"""
compare.py
----------
Runs the full King Domino scoring pipeline on every board image in test_data/
and compares against the corrected ground-truth Excel sheet.

Excel format (sheet "Pointtæller"):
    Header row (Excel row 3):
      Spilleplade | Græs | Skov | Kornmark | Lake | Sump | Mine | Sum | points
    Each Spilleplade row holds:
      - per-terrain TILE COUNTS (Græs..Mine, summing to 24)
      - Sum   = total tile count (should be 24 since the home tile is excluded)
      - points = total score (the value our pipeline tries to match)

Outputs:
    comparison_results.csv             per-board predicted/expected/diff (totals)
    comparison_per_terrain.csv         per-board per-terrain tile counts
    plot_predicted_vs_expected.png     score scatter with y=x
    plot_error_distribution.png        score error histogram
    plot_per_board_diff.png            per-board score error bar chart
    plot_terrain_confusion.png         tile-level terrain confusion-style heatmap
    plot_terrain_counts.png            per-terrain tile counts predicted vs expected
"""

import os
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from joined import terrain_grid, crown_grid, find_clusters
from terrain_classifier import imread_unicode


GROUND_TRUTH_SHEET = "Pointtæller"
GROUND_TRUTH_HEADER_ROW = 3  # 0-indexed row of the header

# Excel terrain columns (Danish, in the order they appear in the sheet).
EXCEL_TERRAINS = ["Græs", "Skov", "Kornmark", "Lake", "Sump", "Mine"]

# English display names for plots.
EN_LABELS = {
    "Græs":     "grass",
    "Skov":     "forest",
    "Kornmark": "wheat",
    "Lake":     "water",
    "Sump":     "swamp",
    "Mine":     "mine",
}

# Map from the SVM's predicted terrain label to the Excel column name.
# Adjust the LEFT-hand keys if your SVM uses different labels.
TERRAIN_LABEL_MAP = {
    "Grassland":  "Græs",
    "Grass":      "Græs",
    "grass":      "Græs",
    "Forest":     "Skov",
    "forest":     "Skov",
    "Field":      "Kornmark",
    "Wheat":      "Kornmark",
    "wheat":      "Kornmark",
    "Lake":       "Lake",
    "Water":      "Lake",
    "water":      "Lake",
    "Swamp":      "Sump",
    "swamp":      "Sump",
    "Mine":       "Mine",
    "mine":       "Mine",
    # Danish passthrough
    "Græs":       "Græs",
    "Skov":       "Skov",
    "Kornmark":   "Kornmark",
    "Sump":       "Sump",
}


# ── Ground truth ──────────────────────────────────────────────────────

def load_ground_truth(xlsx_path):
    """
    Returns:
        totals: dict {board_number: total_points}
        per_terrain_tiles: dict {board_number: {excel_terrain: tile_count}}
    """
    df = pd.read_excel(xlsx_path, sheet_name=GROUND_TRUTH_SHEET,
                       header=GROUND_TRUTH_HEADER_ROW)

    totals = {}
    per_terrain_tiles = {}
    for _, row in df.iterrows():
        label = row.iloc[0]
        if not isinstance(label, str):
            continue
        m = re.match(r"^Spilleplade\s+(\d+)$", label.strip())
        if not m:
            continue
        board_num = int(m.group(1))

        points = row.get("points")
        if pd.isna(points):
            continue
        totals[board_num] = int(points)

        per_terrain_tiles[board_num] = {
            t: int(row[t]) if not pd.isna(row.get(t)) else 0
            for t in EXCEL_TERRAINS
        }
    return totals, per_terrain_tiles


def board_number_from_filename(path):
    stem = Path(path).stem
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


# ── Pipeline (mirrors joined.score_board, returns terrain tile counts too) ─

def run_board(image_path):
    """
    Runs the full pipeline. Returns:
        total_score: int
        terrain_tile_counts: dict {excel_terrain: count}
    """
    image_bgr = imread_unicode(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    board_id = Path(image_path).stem

    t_grid = terrain_grid(image_bgr)
    c_grid = crown_grid(image_gray, board_id)
    clusters = find_clusters(t_grid)

    total_score = 0
    for terrain, cells in clusters:
        total_score += len(cells) * sum(c_grid[r][c] for r, c in cells)

    # Count predicted tiles per terrain across the whole 5x5 grid.
    counts = {t: 0 for t in EXCEL_TERRAINS}
    unknown = set()
    for row in t_grid:
        for label in row:
            mapped = TERRAIN_LABEL_MAP.get(label)
            if mapped is None:
                unknown.add(label)
                continue
            counts[mapped] += 1
    if unknown:
        print(f"  [warn] unmapped terrain labels skipped: {unknown}")

    return total_score, counts


# ── Plots: totals (score) ─────────────────────────────────────────────

def plot_predicted_vs_expected(matched, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    x = matched["expected"].values
    y = matched["predicted"].values
    correct = matched["correct"].values
    ax.scatter(x[correct], y[correct], c="green", label="exact match",
               alpha=0.7, edgecolor="black")
    ax.scatter(x[~correct], y[~correct], c="red", label="mismatch",
               alpha=0.7, edgecolor="black")
    lo = min(x.min(), y.min()) - 5
    hi = max(x.max(), y.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Expected score (ground truth)")
    ax.set_ylabel("Predicted score")
    ax.set_title("Predicted vs. expected total score per board")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_error_distribution(matched, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    diffs = matched["diff"].values
    bins = np.arange(diffs.min() - 1, diffs.max() + 2) - 0.5
    ax.hist(diffs, bins=bins, color="steelblue", edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("Predicted - Expected (score)")
    ax.set_ylabel("Number of boards")
    ax.set_title("Score error distribution")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_per_board_diff(matched, out_path):
    matched = matched.sort_values("board_number")
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(matched)), 4))
    colors = ["green" if d == 0 else "red" for d in matched["diff"]]
    ax.bar(matched["board_number"].astype(str), matched["diff"],
           color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Board number"); ax.set_ylabel("Predicted - Expected (score)")
    ax.set_title("Per-board score error")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ── Plots: terrain ────────────────────────────────────────────────────

def plot_terrain_counts(per_terrain_df, out_path):
    """Total tile count per terrain across all boards: predicted vs expected."""
    pred_totals = [per_terrain_df[f"pred_{t}"].sum() for t in EXCEL_TERRAINS]
    exp_totals  = [per_terrain_df[f"exp_{t}"].sum()  for t in EXCEL_TERRAINS]
    labels = [EN_LABELS[t] for t in EXCEL_TERRAINS]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, exp_totals, width, label="Expected",
           color="lightgray", edgecolor="black")
    ax.bar(x + width/2, pred_totals, width, label="Predicted",
           color="steelblue", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Total tile count across all boards")
    ax.set_title("Tile count per terrain: predicted vs. expected")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for i, (e, p) in enumerate(zip(exp_totals, pred_totals)):
        ax.text(i - width/2, e, str(int(e)), ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, p, str(int(p)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_terrain_confusion(per_terrain_df, out_path):
    """
    Approximate per-terrain confusion at the board-aggregate level.

    True per-board terrain tile counts are taken from the Excel ground truth.
    Predicted per-board counts come from the SVM's tile classifications.

    Since the Excel does not record WHICH tiles are which terrain (only the
    count), we cannot build a true tile-level confusion matrix from this data
    alone. Instead we plot a normalised heatmap of (predicted vs expected)
    per-class deviations: each cell shows how often the SVM mis-attributed
    tiles between class pairs across all boards, computed as
        sum_b max(0, pred_b[i] - exp_b[i]) when the row class is over-counted
    paired against the column class that is under-counted by the same board.
    Diagonal cells show correctly counted tiles.
    """
    n = len(EXCEL_TERRAINS)
    mat = np.zeros((n, n), dtype=int)
    labels = [EN_LABELS[t] for t in EXCEL_TERRAINS]

    for _, row in per_terrain_df.iterrows():
        exp = np.array([row[f"exp_{t}"] for t in EXCEL_TERRAINS])
        pred = np.array([row[f"pred_{t}"] for t in EXCEL_TERRAINS])
        # Diagonal: tiles that the prediction got right (lower bound is min)
        for i in range(n):
            mat[i, i] += min(exp[i], pred[i])
        # Off-diagonal: redistribute the surplus predictions to the deficits
        deficits = np.maximum(exp - pred, 0).astype(float)  # under-predicted
        surpluses = np.maximum(pred - exp, 0).astype(float)  # over-predicted
        d_total = deficits.sum()
        if d_total > 0:
            for j in range(n):  # over-predicted column
                if surpluses[j] == 0:
                    continue
                # distribute this surplus across deficit rows proportionally
                share = deficits / d_total
                for i in range(n):  # true row
                    if i == j:
                        continue
                    mat[i, j] += int(round(surpluses[j] * share[i]))

    # Row-normalise for the right-hand display
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat, dtype=float),
                     where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, fmt in [
        (axes[0], mat, "Tile counts (raw)", "{:d}"),
        (axes[1], norm, "Tile counts (row-normalised)", "{:.2f}"),
    ]:
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                if (isinstance(v, (int, np.integer)) and v != 0) or \
                   (isinstance(v, (float, np.floating)) and abs(v) > 0.005):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            color="white" if (i != j and (data[i, j] < data.max() * 0.6)) else "black",
                            fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Terrain SVM — confusion-style tile-count comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="test_data")
    parser.add_argument("--truth", default="ground_truth.xlsx")
    parser.add_argument("--out", default="comparison_results.csv")
    parser.add_argument("--terrain-out", default="comparison_per_terrain.csv")
    parser.add_argument("--plot-dir", default=".")
    args = parser.parse_args()

    truth_totals, truth_terrain_tiles = load_ground_truth(args.truth)
    print(f"Loaded {len(truth_totals)} ground-truth boards from {args.truth}")

    patterns = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(args.images, pat)))
    image_paths.sort()
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images}")
    print(f"Found {len(image_paths)} images in {args.images}\n")

    score_rows = []
    terrain_rows = []
    for path in image_paths:
        name = os.path.basename(path)
        board_num = board_number_from_filename(path)
        expected_total = truth_totals.get(board_num)
        expected_tiles = truth_terrain_tiles.get(board_num, {})

        print("=" * 70)
        print(f"Board: {name}  (number = {board_num}, expected score = {expected_total})")
        print("=" * 70)

        try:
            predicted_total, predicted_tiles = run_board(path)
            print(f"  Predicted score: {predicted_total}")
            print(f"  Predicted tile counts: {predicted_tiles}")
            if expected_tiles:
                print(f"  Expected  tile counts: {expected_tiles}")
        except Exception as e:
            print(f"  ERROR: {e}")
            predicted_total, predicted_tiles = None, {}

        diff = (predicted_total - expected_total) if (predicted_total is not None
                                                       and expected_total is not None) else None
        correct = (predicted_total == expected_total) if diff is not None else False

        score_rows.append({
            "filename": name,
            "board_number": board_num,
            "predicted": predicted_total,
            "expected": expected_total,
            "diff": diff,
            "correct": correct,
        })

        if board_num is not None and expected_tiles and predicted_total is not None:
            row = {"filename": name, "board_number": board_num}
            for t in EXCEL_TERRAINS:
                row[f"exp_{t}"] = expected_tiles.get(t, 0)
                row[f"pred_{t}"] = predicted_tiles.get(t, 0)
            terrain_rows.append(row)
        print()

    df = pd.DataFrame(score_rows)
    df.to_csv(args.out, index=False)

    matched = df[df["expected"].notna() & df["predicted"].notna()].copy()
    matched["expected"] = matched["expected"].astype(int)
    matched["predicted"] = matched["predicted"].astype(int)
    matched["diff"] = matched["diff"].astype(int)

    n = len(matched)
    n_correct = int(matched["correct"].sum())
    mae = matched["diff"].abs().mean() if n else float("nan")
    rmse = np.sqrt((matched["diff"] ** 2).mean()) if n else float("nan")

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Boards compared:      {n}")
    if n:
        print(f"Exact matches:        {n_correct}/{n}  ({100*n_correct/n:.1f}%)")
        print(f"Mean absolute error:  {mae:.2f}")
        print(f"Root mean sq. error:  {rmse:.2f}")
        print(f"Total predicted:      {int(matched['predicted'].sum())}")
        print(f"Total expected:       {int(matched['expected'].sum())}")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if n:
        plot_predicted_vs_expected(matched, plot_dir / "plot_predicted_vs_expected.png")
        plot_error_distribution(matched, plot_dir / "plot_error_distribution.png")
        plot_per_board_diff(matched, plot_dir / "plot_per_board_diff.png")

    if terrain_rows:
        terrain_df = pd.DataFrame(terrain_rows)
        terrain_df.to_csv(args.terrain_out, index=False)
        plot_terrain_counts(terrain_df, plot_dir / "plot_terrain_counts.png")
        plot_terrain_confusion(terrain_df, plot_dir / "plot_terrain_confusion.png")

        # Per-terrain summary table
        print("\nPer-terrain summary (tile counts summed across boards):")
        print(f"  {'terrain':<10} {'expected':>10} {'predicted':>10} {'diff':>8}")
        for t in EXCEL_TERRAINS:
            e = int(terrain_df[f"exp_{t}"].sum())
            p = int(terrain_df[f"pred_{t}"].sum())
            print(f"  {EN_LABELS[t]:<10} {e:>10} {p:>10} {p-e:>+8}")

    print(f"\nPlots saved to {plot_dir.resolve()}")
    print(f"Saved per-board comparison to {args.out}")
    if terrain_rows:
        print(f"Saved per-terrain comparison to {args.terrain_out}")


if __name__ == "__main__":
    main()