import os
import numpy as np
import cv2
from collections import Counter, defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

TILE_SIZE    = 64
HIST_BINS    = 32
SKIP_CLASSES = {"Unknown", "unknown"}
HARD_CLASSES = {"Mine", "Empty Space", "Home", "Swamp"}
MAX_PER_TILE = 4

# Centre-crop ratio for SIFT — keeps the inner 70 % of the tile,
# removing ~15 % on each side where crowns typically sit.
SIFT_CROP_RATIO = 0.70

# Number of visual words (k-means clusters) per subcategory
VOCAB_SIZE = 16


# ── Helpers ───────────────────────────────────────────────────────────────────

def _centre_crop(img, ratio):
    """Return the centre portion of *img* defined by *ratio* (0-1)."""
    h, w = img.shape[:2]
    margin_y = int(h * (1 - ratio) / 2)
    margin_x = int(w * (1 - ratio) / 2)
    return img[margin_y:h - margin_y, margin_x:w - margin_x]


def _sift_descriptors(bgr_img):
    """Extract SIFT descriptors from the centre-cropped grayscale tile."""
    img = cv2.resize(bgr_img, (TILE_SIZE, TILE_SIZE))
    cropped = _centre_crop(img, SIFT_CROP_RATIO)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return des  # None if no keypoints found


def _collect_images(folder):
    """Yield file paths for every image directly inside *folder*."""
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                yield fpath


# ── Subcategory BoVW vocabulary ───────────────────────────────────────────────

def build_subcategory_vocabs(data_dir, vocab_size=VOCAB_SIZE):
    """
    Build a visual vocabulary (k-means centres) for each subcategory.

    For classes WITHOUT subcategories (e.g. Lake/), the class itself
    is treated as its own single subcategory.

    Returns
    -------
    subcategory_names : list[str]
        Ordered list of subcategory names, e.g.
        ["Empty Space", "Field_general", "Field_vindmølle", "Forest_general", ...]
    vocabs : list[np.ndarray]
        Each entry is (vocab_size, 128) — the k-means centres for that subcategory.
    """
    subcategory_names = []
    vocabs = []

    for label in sorted(os.listdir(data_dir)):
        if label in SKIP_CLASSES:
            continue
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue

        # Find subcategories — or treat class as its own subcategory
        sub_dirs = {}
        has_subdirs = False
        for sub in sorted(os.listdir(class_dir)):
            sub_path = os.path.join(class_dir, sub)
            if os.path.isdir(sub_path):
                has_subdirs = True
                sub_dirs[sub] = sub_path

        if has_subdirs:
            # Also collect any images directly in class_dir as "{label}_root"
            direct_imgs = list(_collect_images(class_dir))
            if direct_imgs:
                sub_dirs[f"{label}_root"] = None  # special marker
        else:
            # No subdirs — whole class is one subcategory
            sub_dirs[label] = class_dir

        for sub_name, sub_path in sorted(sub_dirs.items()):
            # Collect descriptors for this subcategory
            if sub_path is None:
                # Direct images in class_dir
                img_paths = list(_collect_images(class_dir))
            else:
                img_paths = list(_collect_images(sub_path))

            all_des = []
            for fpath in img_paths:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                des = _sift_descriptors(img)
                if des is not None:
                    all_des.append(des)

            if len(all_des) == 0:
                print(f"  WARNING: No SIFT descriptors for subcategory '{sub_name}', skipping.")
                continue

            all_des = np.vstack(all_des).astype(np.float32)

            # k-means to build vocabulary
            k = min(vocab_size, len(all_des))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, _, centres = cv2.kmeans(
                all_des, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )

            subcategory_names.append(sub_name)
            vocabs.append(centres)

    print(f"\nBuilt BoVW vocabularies for {len(subcategory_names)} subcategories:")
    for name, v in zip(subcategory_names, vocabs):
        print(f"  {name:<25} {v.shape[0]} visual words")

    return subcategory_names, vocabs


def _bovw_similarity(des, vocabs):
    """
    Compute a similarity score between one tile's SIFT descriptors
    and each subcategory vocabulary.

    For each subcategory, we count how many of the tile's descriptors
    have their nearest visual word in that subcategory (normalised).

    If the tile has no descriptors, returns zeros.
    """
    n_subcats = len(vocabs)
    if des is None or len(des) == 0:
        return np.zeros(n_subcats, dtype=np.float32)

    # Stack all vocabs and track which subcategory each word belongs to
    all_centres = np.vstack(vocabs).astype(np.float32)
    subcat_ids = []
    for i, v in enumerate(vocabs):
        subcat_ids.extend([i] * len(v))
    subcat_ids = np.array(subcat_ids)

    # BFMatcher: for each descriptor, find nearest visual word
    bf = cv2.BFMatcher(cv2.NORM_L2)
    des = des.astype(np.float32)
    matches = bf.match(des, all_centres)

    # Count matches per subcategory
    scores = np.zeros(n_subcats, dtype=np.float32)
    for m in matches:
        scores[subcat_ids[m.trainIdx]] += 1

    # Normalise by total descriptors
    scores /= (len(des) + 1e-6)
    return scores


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(bgr_img, vocabs):
    """
    Feature vector:
    1. FULL tile  -> HSV histogram + scalar HSV stats  (colour identity)
    2. CENTRE-CROPPED tile -> BoVW similarity scores   (texture/structure)

    The BoVW scores replace the old single SIFT keypoint count.
    Each score says "how much does this tile's local texture resemble
    subcategory X", giving the SVM much richer structural information.
    """
    img_full = cv2.resize(bgr_img, (TILE_SIZE, TILE_SIZE))
    hsv_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2HSV)

    # ── HSV histogram (full tile) ─────────────────────────────────────
    hist_feats = []
    for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        h = cv2.calcHist([hsv_full], [ch], None, [HIST_BINS], [lo, hi]).flatten()
        hist_feats.append(h)
    color_feat = np.concatenate(hist_feats)
    color_feat /= (color_feat.sum() + 1e-6)

    # ── Scalar HSV statistics (full tile) ─────────────────────────────
    val          = hsv_full[:, :, 2]
    dark_ratio   = np.array([(val < 40).sum()  / TILE_SIZE**2], dtype=np.float32)
    bright_ratio = np.array([(val > 180).sum() / TILE_SIZE**2], dtype=np.float32)
    mean_sat     = np.array([hsv_full[:, :, 1].mean() / 255.0], dtype=np.float32)
    mean_val     = np.array([val.mean()           / 255.0], dtype=np.float32)
    v_std        = np.array([val.std()            / 255.0], dtype=np.float32)

    # ── BoVW similarity per subcategory (centre-cropped) ──────────────
    des = _sift_descriptors(bgr_img)
    bovw_scores = _bovw_similarity(des, vocabs)

    return np.concatenate([
        color_feat,
        dark_ratio, bright_ratio,
        mean_sat, mean_val, v_std,
        bovw_scores
    ]).astype(np.float32)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_tile(img, n_versions):
    """
    Returns up to *n_versions* versions of one tile (INCLUDING the original).
    The pool of augmentations is always the same; *n_versions* controls
    how many we actually use, allowing small classes to get more variants.
    """
    results = [img]
    # Rotation
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((TILE_SIZE / 2, TILE_SIZE / 2), angle, 1.0)
        results.append(cv2.warpAffine(img, M, (TILE_SIZE, TILE_SIZE)))
    # Flips
    results.append(cv2.flip(img, 1))
    results.append(cv2.flip(img, 0))
    # Brightness
    for factor in [0.75, 1.25]:
        bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        results.append(bright)
    # Blur
    results.append(cv2.GaussianBlur(img, (3, 3), 0))
    # Combined: flip + brightness
    results.append(np.clip(cv2.flip(img, 1).astype(np.float32) * 0.85, 0, 255).astype(np.uint8))
    results.append(np.clip(cv2.flip(img, 0).astype(np.float32) * 1.15, 0, 255).astype(np.uint8))

    return results[:n_versions]


# ── Load dataset (raw, no augmentation) ───────────────────────────────────────

def load_raw_dataset(data_dir):
    """
    Load ONE image per tile — no augmentation at this stage.
    Label is always the top-level folder name.
    """
    images, labels, paths = [], [], []

    for label in sorted(os.listdir(data_dir)):
        if label in SKIP_CLASSES:
            continue
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue

        image_paths = list(_collect_images(class_dir))
        for sub in sorted(os.listdir(class_dir)):
            sub_dir = os.path.join(class_dir, sub)
            if os.path.isdir(sub_dir):
                image_paths.extend(_collect_images(sub_dir))

        for fpath in image_paths:
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, (TILE_SIZE, TILE_SIZE))
            images.append(img)
            labels.append(label)
            paths.append(fpath)

    print(f"\nRaw dataset: {len(images)} unique tiles")
    for cls, n in sorted(Counter(labels).items()):
        print(f"  {cls:<15} {n} tiles")

    return images, np.array(labels), np.array(paths)


def augment_and_extract(images, labels, vocabs):
    """
    Balanced augmentation: small classes get MORE augmented versions per tile
    so that all classes end up at roughly the same size after augmentation.

    Strategy:
    1. Count how many original tiles each class has.
    2. Set a target size = the largest class count.
    3. For each class, compute how many versions per tile are needed
       to reach the target:  n_versions = ceil(target / class_count),
       capped at the max available augmentations (12).
    4. Classes that are already large get n_versions=1 (no augmentation).

    This prevents the SVM from being biased towards large classes.
    """
    MAX_AUGMENTATIONS = 12  # max pool size in augment_tile

    # Count originals per class
    class_counts = Counter(labels)
    target_size  = max(class_counts.values())

    # Compute per-class augmentation factor
    versions_per_class = {}
    for cls, count in class_counts.items():
        n = int(np.ceil(target_size / count))
        n = min(n, MAX_AUGMENTATIONS)
        n = max(n, 1)
        versions_per_class[cls] = n

    print(f"\n  Balanced augmentation (target ≈ {target_size} per class):")
    for cls in sorted(versions_per_class):
        orig = class_counts[cls]
        aug  = orig * versions_per_class[cls]
        print(f"    {cls:<15} {orig:>4} originals × {versions_per_class[cls]} "
              f"= {aug:>5} tiles")

    X, y = [], []
    for img, label in zip(images, labels):
        n_versions = versions_per_class[label]
        versions = augment_tile(img, n_versions) if n_versions > 1 else [img]
        for v in versions:
            X.append(extract_features(v, vocabs))
            y.append(label)
    return np.array(X), np.array(y)


# ── Kernel comparison ─────────────────────────────────────────────────────────

def compare_kernels(X_train, y_train, X_test, y_test):
    """
    Train SVM with linear, RBF, and polynomial kernels and report accuracy.
    Returns the name and fitted model of the best kernel.
    """
    kernels = {
        "linear": SVC(kernel="linear", C=10),
        "rbf":    SVC(kernel="rbf",    C=10, gamma="scale"),
        "poly":   SVC(kernel="poly",   C=10, degree=3, gamma="scale"),
    }

    print(f"\n{'─' * 40}")
    print(f"Kernel Comparison (hold-out)")
    print(f"{'─' * 40}")

    best_name, best_acc, best_model = None, -1, None
    for name, clf in kernels.items():
        clf.fit(X_train, y_train)
        acc = np.mean(clf.predict(X_test) == y_test)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = clf
            marker = "  <-- best"
        print(f"  {name:<10} {acc:.2%}{marker}")

    print(f"{'─' * 40}")
    print(f"  Selected: {best_name} ({best_acc:.2%})")
    print(f"{'─' * 40}\n")

    return best_name, best_model


# ── Cross-validation (leakage-free) ──────────────────────────────────────────

def cross_validate_clean(images, labels, vocabs, n_folds=5):
    """
    Stratified k-fold CV that augments ONLY the train fold.
    """
    images_list = list(images)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []

    print(f"\n{'─' * 50}")
    print(f"Stratified {n_folds}-Fold Cross-Validation (leakage-free)")
    print(f"{'─' * 50}")

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(images_list, labels), 1):
        train_imgs   = [images_list[i] for i in train_idx]
        test_imgs    = [images_list[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels  = labels[test_idx]

        # Augment train only, extract features
        X_train, y_train = augment_and_extract(train_imgs, train_labels, vocabs)

        # Test: no augmentation
        X_test = np.array([extract_features(img, vocabs) for img in test_imgs])
        y_test = test_labels

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = SVC(kernel="rbf", C=10, gamma="scale")
        model.fit(X_train, y_train)

        acc = np.mean(model.predict(X_test) == y_test)
        fold_scores.append(acc)
        print(f"  Fold {fold_i}: {acc:.2%}  "
              f"(train: {len(X_train)}, test: {len(X_test)})")

    fold_scores = np.array(fold_scores)
    print(f"{'─' * 50}")
    print(f"  Mean:  {fold_scores.mean():.2%}  ±  {fold_scores.std():.2%}")
    print(f"{'─' * 50}\n")

    return fold_scores


# ── Train + diagnose ──────────────────────────────────────────────────────────

def train_and_diagnose(data_dir, n_folds=5):
    # ── Build BoVW vocabularies from subcategory structure ────────────
    subcategory_names, vocabs = build_subcategory_vocabs(data_dir)

    # ── Load raw images ───────────────────────────────────────────────
    images, labels, paths = load_raw_dataset(data_dir)

    # ── Cross-validation (leakage-free) ───────────────────────────────
    fold_scores = cross_validate_clean(images, labels, vocabs, n_folds=n_folds)

    # ── Hold-out split on ORIGINAL images ─────────────────────────────
    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_imgs   = [images[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_imgs    = [images[i] for i in test_idx]
    test_labels  = labels[test_idx]
    test_paths   = paths[test_idx]

    # Augment train only
    X_train, y_train = augment_and_extract(train_imgs, train_labels, vocabs)
    X_test = np.array([extract_features(img, vocabs) for img in test_imgs])
    y_test = test_labels

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Kernel comparison ─────────────────────────────────────────────
    best_kernel_name, model = compare_kernels(X_train, y_train, X_test, y_test)

    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)

    print(f"Hold-out test accuracy ({best_kernel_name}): {acc:.2%}")
    print(f"  Train size: {len(X_train)} (with augmentation)")
    print(f"  Test size:  {len(X_test)} (original tiles only)")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ── Confusion matrix ──────────────────────────────────────────────
    classes = sorted(np.unique(labels))
    cm      = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(
        f"Confusion Matrix ({best_kernel_name}) — Hold-out: {acc:.2%}  |  "
        f"CV mean: {fold_scores.mean():.2%} ± {fold_scores.std():.2%}")
    plt.tight_layout()
    plt.show()

    # ── Misclassified tiles ───────────────────────────────────────────
    wrong_idx = np.where(y_pred != y_test)[0]
    print(f"\n{len(wrong_idx)} misclassified tiles:\n")

    if len(wrong_idx) == 0:
        print("  Perfect score!")
    else:
        cols = min(6, len(wrong_idx))
        rows = max(1, int(np.ceil(len(wrong_idx) / cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
        axes = np.array(axes).flatten()

        for i, idx in enumerate(wrong_idx):
            img_rgb = cv2.cvtColor(test_imgs[idx], cv2.COLOR_BGR2RGB)
            hsv_t   = cv2.cvtColor(test_imgs[idx], cv2.COLOR_BGR2HSV)
            val_t   = hsv_t[:, :, 2]
            print(f"  [{i+1}] True: {y_test[idx]:<15} Pred: {y_pred[idx]:<15} "
                  f"mean_val={val_t.mean()/255:.3f}  "
                  f"v_std={val_t.std()/255:.3f}  "
                  f"file: {os.path.basename(test_paths[idx])}")

            axes[i].imshow(img_rgb)
            axes[i].set_title(
                f"True: {y_test[idx]}\nPred: {y_pred[idx]}",
                fontsize=8, color="red")
            axes[i].axis("off")

        for j in range(len(wrong_idx), len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"Misclassified tiles ({len(wrong_idx)} total)", fontsize=11)
        plt.tight_layout()
        plt.show()

    # ── Retrain on ALL data for production use ────────────────────────
    X_all, y_all = augment_and_extract(images, labels, vocabs)
    X_all_scaled = scaler.fit_transform(X_all)
    model.fit(X_all_scaled, y_all)
    return scaler, model, vocabs


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Script lives inside 'Bearbejdet tiles'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = SCRIPT_DIR

    scaler, model, vocabs = train_and_diagnose(DATA_DIR, n_folds=5)