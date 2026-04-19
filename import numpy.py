import numpy
import joblib
from detect import detect_on_board, slice_board
from Prototype2_5_endelig_model import extract_features
scaler = joblib.load( "model/scaler.joblib")
svmterrain = joblib.load( "model/svm_model.joblib")
subclass_map = joblib.load( "model/subclass_map.joblib")


def classify_tile(tile_bgr):
        vector = extract_features(tile_bgr).reshape(1, -1)
        scaled = scaler.transform(vector)
        prediction = svmterrain.predict(scaled)
        subclass = prediction[0]
        intel = subclass_map[subclass]
        return intel 

def terrain_grid(image):
    grid = [[None]*5 for _ in range(5)]
    for info  in slice_board(image):
          row, col, tile = info
          label = classify_tile(tile)
          grid[row][col] = label
    return grid

def crown_grid(image):
    gridc = numpy.zeros((5, 5), dtype=int)
    detections = detect_on_board(image)
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
    up = flood_fill(row - 1, col, terrain, terrain_grid, visited)
    down = flood_fill(row + 1, col, terrain, terrain_grid, visited)
    left = flood_fill(row , col - 1, terrain, terrain_grid, visited)
    right = flood_fill(row , col + 1, terrain, terrain_grid, visited)
    return [(row, col)] + up + down + left + right


def find_clusters(terrain_grid):
    visited = [[False]*5 for _ in range(5)]
    clusters = []
    
    for row in range(5):
        for col in range(5):
            if not visited[row][col]:
                terrain = terrain_grid[row][col]
                cluster = flood_fill(row, col, terrain, terrain_grid, visited)
                clusters.append(cluster)
    
    return clusters

def score_cluster(cluster, crown_grid):
    tile_count = len(cluster)
    crown_sum = sum(crown_grid[row][col] for row, col in cluster)
    return tile_count * crown_sum

import cv2

def score_board(image_path):
    image = cv2.imread(image_path)
    t_grid = terrain_grid(image)
    c_grid = crown_grid(image)
    clusters = find_clusters(t_grid)
    total = sum(score_cluster(cluster, c_grid) for cluster in clusters)
    print("Terrain:", t_grid)
    print("Crowns:\n", c_grid)
    print("Clusters:", clusters)
    print("Score:", total)
    return total

if __name__ == "__main__":
    score_board("path/to/a/board.jpg")