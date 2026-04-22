import os
import numpy as np
import cv2
import joblib
from pathlib import Path
from detect import detect_on_board, slice_board
from terrain_classifier import extract_features, imread_unicode

# ── Load trained artifacts ────────────────────────────────────────────
# Terrain classifier: vocabs.joblib contains {"class_names": [...], "vocabs": [...]}.
scaler       = joblib.load("model/scaler.joblib")
svmterrain   = joblib.load("model/svm_model.joblib")
vocab_bundle = joblib.load("model/vocabs.joblib")
vocabs       = vocab_bundle["vocabs"]

# Crown detector: separate SVM, trained on HOG features, grayscale input.
svmcrown = joblib.load("svm_model.pkl")

# Terrain types that exist on the board but do not form scorable clusters.
NON_SCORING_TERRAINS = {"Empty Space", "Home"}

def classify_tile(tile_bgr):
    vector = extract_features(tile_bgr, vocabs).reshape(1, -1)
    scaled = scaler.transform(vector)
    prediction = svmterrain.predict(scaled)
    return prediction[0]


def terrain_grid(image):
    grid = [[None] * 5 for _ in range(5)]
    for info in slice_board(image):
        row, col, tile = info
        grid[row][col] = classify_tile(tile)
    return grid


def crown_grid(image_gray, board_id):
    gridc = np.zeros((5, 5), dtype=int)
    detections = detect_on_board(image_gray, svmcrown, board_id)
    for record in detections:
        gridc[record["tile_row"]][record["tile_col"]] += 1
    return gridc


def flood_fill(row, col, terrain, terrain_grid, visited):
    if row < 0 or row > 4:
        return []
    if col < 0 or col > 4:
        return []
    if visited[row][col]:
        return []
    if terrain_grid[row][col] != terrain:
        return []
    visited[row][col] = True
    up    = flood_fill(row - 1, col, terrain, terrain_grid, visited)
    down  = flood_fill(row + 1, col, terrain, terrain_grid, visited)
    left  = flood_fill(row, col - 1, terrain, terrain_grid, visited)
    right = flood_fill(row, col + 1, terrain, terrain_grid, visited)
    return [(row, col)] + up + down + left + right


def find_clusters(terrain_grid):
    visited = [[False] * 5 for _ in range(5)]
    clusters = []

    for row in range(5):
        for col in range(5):
            if visited[row][col]:
                continue
            terrain = terrain_grid[row][col]
            cluster = flood_fill(row, col, terrain, terrain_grid, visited)
            # Skip non-scoring terrains (Empty Space, Home).
            if terrain in NON_SCORING_TERRAINS:
                continue
            clusters.append((terrain, cluster))

    return clusters


def score_cluster(cluster, crown_grid):
    tile_count = len(cluster)
    crown_sum = sum(crown_grid[row][col] for row, col in cluster)
    return tile_count * crown_sum


def score_board(image_path):
    # Colour version for terrain classifier
    image_bgr = imread_unicode(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Grayscale version for crown detector
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    board_id = Path(image_path).stem

    t_grid = terrain_grid(image_bgr)
    c_grid = crown_grid(image_gray, board_id)
    clusters = find_clusters(t_grid)

    total = sum(score_cluster(cells, c_grid) for _terrain, cells in clusters)

    print("Terrain grid:")
    for row in t_grid:
        print(" ", row)
    print("\nCrown grid:")
    print(c_grid)
    print("\nClusters (terrain, cells):")
    for terrain, cells in clusters:
        cluster_score = len(cells) * sum(c_grid[r][c] for r, c in cells)
        print(f"  {terrain:<12} {len(cells)} tiles  score={cluster_score}")
    print(f"\nTotal score: {total}")
    return total


if __name__ == "__main__":
    import os
    import glob

    VALIDATION_DIR = r"C:\exercices\miniproject\KingDominoMK2\pictures\validation_data"

    # Collect all image files in the folder
    patterns = ("*.jpg", "*.jpeg", "*.png")
    board_paths = []
    for pat in patterns:
        board_paths.extend(glob.glob(os.path.join(VALIDATION_DIR, pat)))
    board_paths.sort()

    if not board_paths:
        raise FileNotFoundError(f"No images found in {VALIDATION_DIR}")

    print(f"Found {len(board_paths)} boards.\n")

    results = []
    for path in board_paths:
        print("=" * 70)
        print(f"Board: {os.path.basename(path)}")
        print("=" * 70)
        try:
            score = score_board(path)
            results.append((os.path.basename(path), score))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((os.path.basename(path), None))
        print()

    # Summary at the end
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, score in results:
        score_str = f"{score}" if score is not None else "ERROR"
        print(f"  {name:<40} {score_str}")