"""
Save hyperparameters of NYYYY_KINGDOMINO_YALLAH_3.py to a pickle file.

Hardcoded paths — just double-click to run.
"""

import pickle
from pathlib import Path


# ── Hyperparameters extracted from NYYYY_KINGDOMINO_YALLAH_3.py ──────────────

hyperparams = {
    # ── Tile / image settings ────────────────────────────────────────────
    "tile_size": 64,
    "hist_bins": 32,
    "skip_classes": {"Unknown", "unknown"},
    "hard_classes": {"Mine", "Empty Space", "Home", "Swamp"},
    "max_per_tile": 4,

    # ── SIFT / BoVW settings ─────────────────────────────────────────────
    "sift_crop_ratio": 0.50,      # centre-crop for SIFT (inner 50 % of tile)
    "vocab_size": 24,             # k-means clusters per per-class vocabulary
    "kmeans_criteria_max_iter": 100,
    "kmeans_criteria_eps": 0.2,
    "kmeans_attempts": 10,
    "kmeans_init_flag": "KMEANS_PP_CENTERS",

    # ── HSV feature settings ─────────────────────────────────────────────
    "hsv_channel_ranges": [(0, 180), (0, 256), (0, 256)],  # H, S, V
    "dark_val_threshold": 40,     # V < 40 -> "dark" pixel
    "bright_val_threshold": 180,  # V > 180 -> "bright" pixel

    # ── Augmentation settings ────────────────────────────────────────────
    "max_augmentations": 12,      # max pool size in augment_tile
    "rotation_angles": [90, 180, 270],
    "flip_codes": [1, 0],         # 1 = horizontal, 0 = vertical
    "brightness_factors": [0.75, 1.25],
    "gaussian_blur_ksize": (3, 3),
    "gaussian_blur_sigma": 0,
    "combined_aug_brightness": [0.85, 1.15],

    # ── SVM hyperparameters (compare_kernels) ────────────────────────────
    "svm_kernels": {
        "linear": {"kernel": "linear", "C": 10},
        "rbf":    {"kernel": "rbf",    "C": 10, "gamma": "scale"},
        "poly":   {"kernel": "poly",   "C": 10, "degree": 3, "gamma": "scale"},
    },

    # ── CV SVM (cross_validate_clean) ────────────────────────────────────
    "cv_svm": {"kernel": "rbf", "C": 10, "gamma": "scale"},

    # ── Data split / CV settings ─────────────────────────────────────────
    "n_folds": 5,
    "test_size": 0.20,
    "random_state": 42,
    "stratified": True,
    "cv_shuffle": True,

    # ── Diagnostic thresholds ────────────────────────────────────────────
    "gap_warn_threshold": 0.10,         # per-fold "⚠️ large gap" marker
    "mean_gap_healthy": 0.05,
    "mean_gap_some_overfit": 0.15,
}


# ── Write pickle via pathlib ─────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
out_path = SCRIPT_DIR / "kingdomino_hyperparams.pkl"

with out_path.open("wb") as f:
    pickle.dump(hyperparams, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Wrote {len(hyperparams)} hyperparameter entries to:")
print(f"  {out_path}")
print(f"  size: {out_path.stat().st_size} bytes")

# ── Sanity check: load it back ───────────────────────────────────────────────

with out_path.open("rb") as f:
    loaded = pickle.load(f)

assert loaded == hyperparams, "Pickle round-trip failed!"
print("Round-trip load OK.")
