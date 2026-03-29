import os
import cv2
import re
import random
import shutil
from KINDOMINOP0 import get_terrain

# Get the folder where this script is located
folder = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(folder, "Kingdomino_dataen")

val_folder = os.path.join(folder, "validation_data")
os.makedirs(val_folder, exist_ok=True)

files = os.listdir(data_folder)
val_files = random.sample(files, int(len(files) * 0.2))

for f in val_files:
    shutil.move(os.path.join(data_folder, f),
                os.path.join(val_folder, f))

# Create a list of all image files left in the training folder
pictures = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]

# sørg for tiles-mappen findes
tiles_folder = os.path.join(folder, "tiles")
if os.path.exists(tiles_folder):
    shutil.rmtree(tiles_folder)
os.makedirs(tiles_folder, exist_ok=True)

image_size = 100

for picture_path in pictures:
    img = cv2.imread(picture_path)
    if img is None:
        print(f"Failed to load: {picture_path}")
        continue

    base_name = os.path.splitext(os.path.basename(picture_path))[0]

    for row, height in enumerate(range(0, 500, image_size)):
        for col, length in enumerate(range(0, 500, image_size)):
            tile = img[height:height+image_size, length:length+image_size]

            output_path = os.path.join(
                tiles_folder, f"{base_name}_{row}_{col}.jpg")
            cv2.imwrite(output_path, tile)

print("Tiles saved to tiles/ folder")

# --------------------------
# SORTERING
# --------------------------

output_folder = os.path.join(folder, "sorted_tiles")
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Matcher fx: 1_0_0.jpg, 74_4_3.jpeg osv.
tile_pattern = re.compile(r"^\d+_[0-4]_[0-4]\.(jpg|jpeg|png)$", re.IGNORECASE)

for filename in os.listdir(tiles_folder):
    if not tile_pattern.match(filename):
        continue

    path = os.path.join(tiles_folder, filename)
    tile = cv2.imread(path)

    if tile is None:
        print(f"Failed to read tile: {path}")
        continue

    terrain = get_terrain(tile)

    if terrain is None or str(terrain).strip() == "":
        terrain = "UNKNOWN"

    terrain_folder = os.path.join(output_folder, str(terrain))
    os.makedirs(terrain_folder, exist_ok=True)

    save_path = os.path.join(terrain_folder, filename)
    cv2.imwrite(save_path, tile)

print("Done")
