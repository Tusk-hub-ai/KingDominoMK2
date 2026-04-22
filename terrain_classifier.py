import os
import joblib
import numpy as np
import cv2
from collections import Counter, defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

TILE_SIZE = 64
HIST_BINS = 32
SKIP_CLASSES = {"Unknown", "unknown"}
HARD_CLASSES = {"Mine", "Empty Space", "Home", "Swamp"}
MAX_PER_TILE = 4

# Centre-crop ratio for SIFT — keeps the inner 70 % of the tile,
# removing ~15 % on each side where crowns typically sit.
SIFT_CROP_RATIO = 0.50

# Number of visual words (k-means clusters) per tile-type vocabulary
VOCAB_SIZE = 24


# ── Helpers ───────────────────────────────────────────────────────────────────

def imread_unicode(path):
    """cv2.imread wrapper that handles non-ASCII paths (e.g. Danish characters)."""
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img


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


# ── Per-class BoVW vocabulary ────────────────────────────────────────────────

def _collect_images_recursive(folder):
    """
    Yield file paths for every image inside *folder*, including
    all images in any nested subdirectories.
    """
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(root, fname)
                if os.path.isfile(fpath):
                    yield fpath


def build_class_vocabs_from_arrays(images, labels, vocab_size=VOCAB_SIZE):
    """
    Build ONE visual vocabulary (k-means centres) per TILE TYPE from
    in-memory arrays of images and their labels.

    This is the leakage-free version: it only uses the images you pass in,
    so you can call it on a TRAIN fold only.

    Parameters
    ----------
    images : list of BGR np.ndarrays
    labels : np.ndarray of strings (same length as images)

    Returns
    -------
    class_names : list[str]
        Ordered list of tile-type names present in `labels`.
    vocabs : list[np.ndarray]
        Each entry is (vocab_size, 128) — the k-means centres for that class.
    """
    class_names = []
    vocabs = []

    # Group images by class label
    by_class = defaultdict(list)
    for img, lbl in zip(images, labels):
        by_class[lbl].append(img)

    for label in sorted(by_class.keys()):
        if label in SKIP_CLASSES:
            continue

        all_des = []
        for img in by_class[label]:
            des = _sift_descriptors(img)
            if des is not None:
                all_des.append(des)

        if len(all_des) == 0:
            print(
                f"  WARNING: No SIFT descriptors for class '{label}', skipping.")
            continue

        all_des = np.vstack(all_des).astype(np.float32)

        # k-means to build vocabulary
        k = min(vocab_size, len(all_des))
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centres = cv2.kmeans(
            all_des, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        class_names.append(label)
        vocabs.append(centres)

    return class_names, vocabs


def build_class_vocabs(data_dir, vocab_size=VOCAB_SIZE):
    """
    Build ONE visual vocabulary (k-means centres) per TILE TYPE (top-level class).

    Convenience wrapper: reads all images from disk and then delegates
    to build_class_vocabs_from_arrays. Use this if you need a vocabulary
    built on the WHOLE dataset (NOT leakage-free for CV purposes).

    For CV and hold-out training, prefer build_class_vocabs_from_arrays
    called with ONLY the training fold.
    """
    class_names = []
    vocabs = []

    for label in sorted(os.listdir(data_dir)):
        if label in SKIP_CLASSES:
            continue
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue

        # Collect ALL images under this class, including nested subdirs
        img_paths = list(_collect_images_recursive(class_dir))

        all_des = []
        for fpath in img_paths:
            img = imread_unicode(fpath)
            if img is None:
                continue
            des = _sift_descriptors(img)
            if des is not None:
                all_des.append(des)

        if len(all_des) == 0:
            print(
                f"  WARNING: No SIFT descriptors for class '{label}', skipping.")
            continue

        all_des = np.vstack(all_des).astype(np.float32)

        # k-means to build vocabulary
        k = min(vocab_size, len(all_des))
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, _, centres = cv2.kmeans(
            all_des, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        class_names.append(label)
        vocabs.append(centres)

    print(f"\nBuilt BoVW vocabularies for {len(class_names)} tile types:")
    for name, v in zip(class_names, vocabs):
        print(f"  {name:<15} {v.shape[0]} visual words")

    return class_names, vocabs


def _bovw_similarity(des, vocabs):
    """
    Compute a similarity score between one tile's SIFT descriptors
    and each CLASS vocabulary.

    For each class, we count how many of the tile's descriptors
    have their nearest visual word in that class's vocabulary (normalised).

    If the tile has no descriptors, returns zeros.
    """
    n_classes = len(vocabs)
    if des is None or len(des) == 0:
        return np.zeros(n_classes, dtype=np.float32)

    # Stack all vocabs and track which class each word belongs to
    all_centres = np.vstack(vocabs).astype(np.float32)
    class_ids = []
    for i, v in enumerate(vocabs):
        class_ids.extend([i] * len(v))
    class_ids = np.array(class_ids)

    # BFMatcher: for each descriptor, find nearest visual word
    bf = cv2.BFMatcher(cv2.NORM_L2)
    des = des.astype(np.float32)
    matches = bf.match(des, all_centres)

    # Count matches per class
    scores = np.zeros(n_classes, dtype=np.float32)
    for m in matches:
        scores[class_ids[m.trainIdx]] += 1

    # Normalise by total descriptors
    scores /= (len(des) + 1e-6)
    return scores


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(bgr_img, vocabs):
    """
    Feature vector:
    1. FULL tile  -> HSV histogram + scalar HSV stats  (colour identity)
    2. CENTRE-CROPPED tile -> BoVW similarity scores   (texture/structure)

    The BoVW produces one score per TILE TYPE (top-level class).
    Each score says "how much does this tile's local texture resemble
    class X", giving the SVM a compact structural signal that
    matches the label granularity.
    """
    img_full = cv2.resize(bgr_img, (TILE_SIZE, TILE_SIZE))
    hsv_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2HSV)

    # ── HSV histogram (full tile) ─────────────────────────────────────
    hist_feats = []
    for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        h = cv2.calcHist([hsv_full], [ch], None, [
                         HIST_BINS], [lo, hi]).flatten()
        hist_feats.append(h)
    color_feat = np.concatenate(hist_feats)
    color_feat /= (color_feat.sum() + 1e-6)

    # ── Scalar HSV statistics (full tile) ─────────────────────────────
    val = hsv_full[:, :, 2]
    dark_ratio = np.array([(val < 40).sum() / TILE_SIZE**2], dtype=np.float32)
    bright_ratio = np.array(
        [(val > 180).sum() / TILE_SIZE**2], dtype=np.float32)
    median_sat = np.array(
        [np.median(hsv_full[:, :, 1]) / 255.0], dtype=np.float32)
    median_val = np.array([np.median(val) / 255.0], dtype=np.float32)
    v_std = np.array([val.std() / 255.0], dtype=np.float32)

    # ── BoVW similarity per tile-type (centre-cropped) ────────────────
    des = _sift_descriptors(bgr_img)
    bovw_scores = _bovw_similarity(des, vocabs)

    return np.concatenate([
        color_feat,
        dark_ratio, bright_ratio,
        median_sat, median_val, v_std,
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
        bright = np.clip(img.astype(np.float32) *
                         factor, 0, 255).astype(np.uint8)
        results.append(bright)
    # Blur
    results.append(cv2.GaussianBlur(img, (3, 3), 0))
    # Combined: flip + brightness
    results.append(np.clip(cv2.flip(img, 1).astype(
        np.float32) * 0.85, 0, 255).astype(np.uint8))
    results.append(np.clip(cv2.flip(img, 0).astype(
        np.float32) * 1.15, 0, 255).astype(np.uint8))

    return results[:n_versions]


# ── Load dataset (raw, no augmentation) ───────────────────────────────────────

def load_raw_dataset(data_dir):
    """
    Load ONE image per tile — no augmentation at this stage.
    Label is always the top-level folder name.

    All images inside a class folder are loaded, including those
    in any (arbitrarily deep) subdirectories. Subdirectories do NOT
    produce separate labels — everything under Field/ is "Field", etc.
    """
    images, labels, paths = [], [], []

    for label in sorted(os.listdir(data_dir)):
        if label in SKIP_CLASSES:
            continue
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue

        image_paths = list(_collect_images_recursive(class_dir))

        for fpath in image_paths:
            img = imread_unicode(fpath)
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
    target_size = max(class_counts.values())

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
        aug = orig * versions_per_class[cls]
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

def cross_validate_clean(images, labels, n_folds=5):
    """
    Stratified k-fold CV that is FULLY leakage-free:
      - Augmentation happens ONLY on the train fold
      - BoVW vocabulary is built ONLY from the train fold
      - Scaler is fitted ONLY on the train fold

    This gives an honest estimate of how the model generalises to
    completely unseen tiles.
    """
    images_list = list(images)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    train_scores = []  # for overfitting diagnosis

    print(f"\n{'─' * 60}")
    print(f"Stratified {n_folds}-Fold Cross-Validation (FULLY leakage-free)")
    print(f"{'─' * 60}")

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(images_list, labels), 1):
        train_imgs = [images_list[i] for i in train_idx]
        test_imgs = [images_list[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Build vocabulary on THIS fold's train data only (leakage-free)
        _, fold_vocabs = build_class_vocabs_from_arrays(
            train_imgs, train_labels)

        # Augment train only, extract features using fold_vocabs
        X_train, y_train = augment_and_extract(
            train_imgs, train_labels, fold_vocabs)

        # Test: no augmentation, use the SAME fold_vocabs (built only from train)
        X_test = np.array([extract_features(img, fold_vocabs)
                          for img in test_imgs])
        y_test = test_labels

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(kernel="rbf", C=10, gamma="scale")
        model.fit(X_train, y_train)

        train_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(model.predict(X_test) == y_test)
        fold_scores.append(test_acc)
        train_scores.append(train_acc)

        gap = train_acc - test_acc
        gap_marker = "  ⚠️ large gap" if gap > 0.10 else ""
        print(f"  Fold {fold_i}: train={train_acc:.2%}  test={test_acc:.2%}  "
              f"gap={gap:+.2%}{gap_marker}  "
              f"(train: {len(X_train)}, test: {len(X_test)})")

    fold_scores = np.array(fold_scores)
    train_scores = np.array(train_scores)
    mean_gap = train_scores.mean() - fold_scores.mean()
    print(f"{'─' * 60}")
    print(
        f"  Train mean: {train_scores.mean():.2%}  ±  {train_scores.std():.2%}")
    print(
        f"  Test  mean: {fold_scores.mean():.2%}  ±  {fold_scores.std():.2%}")
    print(f"  Mean gap:   {mean_gap:+.2%}  "
          f"{'(healthy)' if mean_gap < 0.05 else '(some overfitting)' if mean_gap < 0.15 else '(OVERFITTING)'}")
    print(f"{'─' * 60}\n")

    return fold_scores


# ── Train + diagnose ──────────────────────────────────────────────────────────

def train_and_diagnose(data_dir, n_folds=5):
    # ── Load raw images (labels = top-level class) ────────────────────
    images, labels, paths = load_raw_dataset(data_dir)

    # ── Cross-validation (FULLY leakage-free: vocab built per fold) ───
    fold_scores = cross_validate_clean(images, labels, n_folds=n_folds)

    # ── Hold-out split on ORIGINAL images ─────────────────────────────
    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_imgs = [images[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_imgs = [images[i] for i in test_idx]
    test_labels = labels[test_idx]
    test_paths = paths[test_idx]

    # Build vocabulary on TRAIN only (leakage-free)
    print("\nBuilding vocabulary on hold-out train split only (leakage-free)...")
    _, holdout_vocabs = build_class_vocabs_from_arrays(
        train_imgs, train_labels)
    print(f"  Built vocabularies for {len(holdout_vocabs)} tile types")

    # Augment train only
    X_train, y_train = augment_and_extract(
        train_imgs, train_labels, holdout_vocabs)
    X_test = np.array([extract_features(img, holdout_vocabs)
                      for img in test_imgs])
    y_test = test_labels

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Kernel comparison ─────────────────────────────────────────────
    best_kernel_name, model = compare_kernels(X_train, y_train, X_test, y_test)

    # Report train vs test accuracy for the chosen model (overfitting check)
    train_acc = np.mean(model.predict(X_train) == y_train)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)

    print(f"Hold-out train accuracy ({best_kernel_name}): {train_acc:.2%}")
    print(f"Hold-out test  accuracy ({best_kernel_name}): {acc:.2%}")
    print(f"Gap (train - test):                   {train_acc - acc:+.2%}")
    print(f"  Train size: {len(X_train)} (with augmentation)")
    print(f"  Test size:  {len(X_test)} (original tiles only)")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ── Confusion matrix ──────────────────────────────────────────────
    classes = sorted(np.unique(labels))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
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
            hsv_t = cv2.cvtColor(test_imgs[idx], cv2.COLOR_BGR2HSV)
            val_t = hsv_t[:, :, 2]
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

        plt.suptitle(
            f"Misclassified tiles ({len(wrong_idx)} total)", fontsize=11)
        plt.tight_layout()
        plt.show()

    # ── Retrain on ALL data for production use ────────────────────────
    # Here it is fine to use all data — this is the final deployed model,
    # not an evaluation. We rebuild the vocabulary on ALL images for the
    # best possible final vocabulary.
    print("\nTraining final production model on ALL data...")
    final_class_names, final_vocabs = build_class_vocabs_from_arrays(images, labels)
    X_all, y_all = augment_and_extract(images, labels, final_vocabs)
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    model.fit(X_all_scaled, y_all)

    # ── Persist artifacts for the integration layer ───────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(final_scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(model,        os.path.join(out_dir, "svm_model.joblib"))
    joblib.dump(
        {"class_names": final_class_names, "vocabs": final_vocabs},
        os.path.join(out_dir, "vocabs.joblib"),
    )

    print(f"\nSaved artifacts to: {out_dir}")
    print(f"  scaler.joblib       ({len(final_scaler.mean_)} feature dims)")
    print(f"  svm_model.joblib    ({model.kernel} kernel)")
    print(f"  vocabs.joblib       ({len(final_class_names)} class vocabularies)")

    return final_scaler, model, final_vocabs
# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Script lives inside 'Bearbejdet tiles'
    if __name__ == "__main__":
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(SCRIPT_DIR, "Alt bearbejdet data")

    scaler, model, vocabs = train_and_diagnose(DATA_DIR, n_folds=5)
