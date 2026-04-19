import os
import numpy as np
import cv2
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

TILE_SIZE    = 64
HIST_BINS    = 32
SKIP_CLASSES = {"Unknown", "unknown"}
AUG_PER_TILE = 4
RANDOM_SEED  = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = r"C:\Users\Jackl\Desktop\Mini projekter\Mini projekt King D"
TRAIN_DIR = os.path.join(BASE_DIR, "sorted_subclassed_tiles")   # subclass folders
TEST_DIR  = os.path.join(BASE_DIR, "Kingdomino_TESTset_tiles")  # sorted into main-class folders

# ── Subclass → main class mapping ─────────────────────────────────────────────
# Built automatically: "Field_general" → "Field", "Field_vindmølle" → "Field"
# The parent folder name IS the main class.

def build_subclass_map(data_dir):
    """Scan sorted_subclassed_tiles and build subclass → main class dict."""
    mapping = {}
    for main_class in sorted(os.listdir(data_dir)):
        main_path = os.path.join(data_dir, main_class)
        if not os.path.isdir(main_path) or main_class in SKIP_CLASSES:
            continue
        for sub in sorted(os.listdir(main_path)):
            sub_path = os.path.join(main_path, sub)
            if os.path.isdir(sub_path):
                mapping[sub] = main_class
        # If no subfolders (images directly in main_class folder), map to self
        if main_class not in mapping.values():
            mapping[main_class] = main_class
    return mapping

# ── SIFT singleton ────────────────────────────────────────────────────────────
_sift = cv2.SIFT_create()

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(bgr_img):
    img = cv2.resize(bgr_img, (TILE_SIZE, TILE_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV histogram
    hist_feats = []
    for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        h = cv2.calcHist([hsv], [ch], None, [HIST_BINS], [lo, hi]).flatten()
        hist_feats.append(h)
    color_feat = np.concatenate(hist_feats)
    color_feat /= (color_feat.sum() + 1e-6)

    # Scalar HSV statistics
    val          = hsv[:, :, 2]
    dark_ratio   = np.array([(val < 40).sum()  / TILE_SIZE**2], dtype=np.float32)
    bright_ratio = np.array([(val > 180).sum() / TILE_SIZE**2], dtype=np.float32)
    mean_sat     = np.array([hsv[:, :, 1].mean() / 255.0],      dtype=np.float32)
    mean_val     = np.array([val.mean()           / 255.0],      dtype=np.float32)
    v_std        = np.array([val.std()            / 255.0],      dtype=np.float32)

    # SIFT keypoint count
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, _ = _sift.detectAndCompute(gray, None)
    sift_count = np.array([len(kp) / 100.0], dtype=np.float32)

    return np.concatenate([
        color_feat,
        dark_ratio, bright_ratio,
        mean_sat, mean_val, v_std,
        sift_count,
    ]).astype(np.float32)

# ── Random augmentation ──────────────────────────────────────────────────────

def random_augment(img, rng):
    h, w = img.shape[:2]
    angle = rng.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    scale = rng.uniform(0.80, 1.0)
    crop_size = int(h * scale)
    y0 = rng.integers(0, h - crop_size + 1)
    x0 = rng.integers(0, w - crop_size + 1)
    img = cv2.resize(img[y0:y0+crop_size, x0:x0+crop_size], (w, h))
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)
    factor = rng.uniform(0.75, 1.25)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img

# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_subclassed_dataset(data_dir):
    """Load from sorted_subclassed_tiles: main_class/subclass/images."""
    images, sub_labels, main_labels = [], [], []
    sub_map = build_subclass_map(data_dir)

    for main_class in sorted(os.listdir(data_dir)):
        main_path = os.path.join(data_dir, main_class)
        if not os.path.isdir(main_path) or main_class in SKIP_CLASSES:
            continue
        for sub in sorted(os.listdir(main_path)):
            sub_path = os.path.join(main_path, sub)
            if not os.path.isdir(sub_path):
                continue
            for fname in os.listdir(sub_path):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(sub_path, fname))
                if img is None:
                    continue
                images.append(cv2.resize(img, (TILE_SIZE, TILE_SIZE)))
                sub_labels.append(sub)
                main_labels.append(main_class)

    return images, np.array(sub_labels), np.array(main_labels)


def load_flat_dataset(data_dir):
    """Load from a flat folder of class/images (for test set)."""
    images, labels = [], []
    for label in sorted(os.listdir(data_dir)):
        if label in SKIP_CLASSES:
            continue
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            images.append(cv2.resize(img, (TILE_SIZE, TILE_SIZE)))
            labels.append(label)
    return images, np.array(labels)

# ── Helpers ───────────────────────────────────────────────────────────────────

def featurise(images):
    return np.array([extract_features(img) for img in images])


def augment_and_featurise(images, labels, rng):
    X, y = [], []
    for img, label in zip(images, labels):
        X.append(extract_features(img))
        y.append(label)
        for _ in range(AUG_PER_TILE):
            X.append(extract_features(random_augment(img, rng)))
            y.append(label)
    return np.array(X), np.array(y)

# ── Train + diagnose ──────────────────────────────────────────────────────────

def train_and_diagnose():
    # ── 1. Load training data (subclasses) ────────────────────────────────
    images, sub_labels, main_labels = load_subclassed_dataset(TRAIN_DIR)
    images = np.array(images)
    sub_map = build_subclass_map(TRAIN_DIR)

    print(f"Training tiles: {len(images)}")
    print(f"Subclasses: {sorted(set(sub_labels))}")
    print(f"Main classes: {sorted(set(main_labels))}")

    # ── 2. Augment training data (train on SUBCLASS labels) ───────────────
    rng = np.random.default_rng(RANDOM_SEED)
    X_train, y_train_sub = augment_and_featurise(images, sub_labels, rng)

    counts = Counter(y_train_sub)
    print(f"\nAfter augmentation: {len(X_train)} tiles")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:<25} {n}")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    # ── 3. Leak-free CV folds ─────────────────────────────────────────────
    n_originals = len(images)
    block_size  = 1 + AUG_PER_TILE
    original_idx = np.arange(n_originals)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    leak_free_folds = []
    for orig_train, orig_test in skf.split(original_idx, sub_labels):
        fold_train = np.concatenate([
            np.arange(i * block_size, (i + 1) * block_size) for i in orig_train
        ])
        fold_test = np.concatenate([
            np.arange(i * block_size, (i + 1) * block_size) for i in orig_test
        ])
        leak_free_folds.append((fold_train, fold_test))

    # ── 4. GridSearchCV on subclass labels ────────────────────────────────
    param_grid = {
        "C":      [0.01, 0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
    }
    grid = GridSearchCV(
        SVC(class_weight="balanced"),
        param_grid,
        cv=leak_free_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_sc, y_train_sub)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV accuracy (subclass): {grid.best_score_:.2%}")
    best_idx = grid.best_index_
    for fold in range(len(leak_free_folds)):
        key = f"split{fold}_test_score"
        print(f"  Fold {fold+1}: {grid.cv_results_[key][best_idx]:.2%}")

    model = grid.best_estimator_

    # ── 5. CV confusion matrix (subclass) ─────────────────────────────────
    y_cv_pred = cross_val_predict(
        SVC(**grid.best_params_, class_weight="balanced"),
        X_train_sc, y_train_sub,
        cv=leak_free_folds, n_jobs=-1,
    )
    classes_sub = sorted(np.unique(y_train_sub))
    cm_cv = confusion_matrix(y_train_sub, y_cv_pred, labels=classes_sub)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm_cv, display_labels=classes_sub).plot(
        ax=ax, cmap="Blues", colorbar=False,
    )
    ax.set_title(f"CV CM (subclass) — acc: {grid.best_score_:.2%}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ── 6. CV confusion matrix mapped to MAIN classes ─────────────────────
    y_cv_main  = np.array([sub_map[s] for s in y_cv_pred])
    y_true_main = np.array([sub_map[s] for s in y_train_sub])
    classes_main = sorted(set(sub_map.values()))
    main_acc = (y_cv_main == y_true_main).mean()

    cm_main = confusion_matrix(y_true_main, y_cv_main, labels=classes_main)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm_main, display_labels=classes_main).plot(
        ax=ax2, cmap="Greens", colorbar=False,
    )
    ax2.set_title(f"CV CM (main class) — acc: {main_acc:.2%}")
    plt.tight_layout()
    plt.show()

    # ── 7. Held-out test (TESTset tiles, sorted into main-class folders) ──
    if os.path.isdir(TEST_DIR):
        test_images, test_labels = load_flat_dataset(TEST_DIR)
        if len(test_images) > 0:
            X_test    = featurise(test_images)
            X_test_sc = scaler.transform(X_test)

            # Predict subclass, map to main
            y_test_sub  = model.predict(X_test_sc)
            y_test_pred = np.array([sub_map.get(s, s) for s in y_test_sub])

            test_acc = (y_test_pred == test_labels).mean()
            print(f"\nHeld-out test accuracy (main class): {test_acc:.2%}")

            cm_test = confusion_matrix(test_labels, y_test_pred, labels=classes_main)
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(cm_test, display_labels=classes_main).plot(
                ax=ax3, cmap="Oranges", colorbar=False,
            )
            ax3.set_title(f"Held-out test CM — acc: {test_acc:.2%}")
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo test tiles found — sort TESTset_tiles into class folders first.")
    else:
        print(f"\nTest dir not found: {TEST_DIR}")
        print("Run tile_testset.py first, then sort tiles into class folders.")

    # ── Save model ────────────────────────────────────────────────────────
    save_dir = os.path.join(BASE_DIR, "model")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler,  os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(model,   os.path.join(save_dir, "svm_model.joblib"))
    joblib.dump(sub_map, os.path.join(save_dir, "subclass_map.joblib"))
    print(f"\nModel saved to {save_dir}")

    return scaler, model, sub_map


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scaler, model, sub_map = train_and_diagnose()
