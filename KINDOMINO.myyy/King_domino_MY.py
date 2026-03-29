import os
import cv2
import re
from KINDOMINOP0 import get_terrain

# Get the folder where this script is located

folder = os.path.dirname(os.path.abspath(__file__))

data_folder = os.path.join(folder, "Kingdomino_dataen")

# Create a list of all image files (1.jpg to 74.jpg) in that folder
pictures = [os.path.join(data_folder, f"{i}.jpg") for i in range(1, 75)]


# ✅ sørg for tiles-mappen findes
tiles_folder = os.path.join(folder, "tiles")
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

            # ✅ GEM I tiles/ i stedet for main
            output_path = os.path.join(
                tiles_folder, f"{base_name}_{row}_{col}.jpg")
            cv2.imwrite(output_path, tile)

print("Tiles saved to tiles/ folder")
# --------------------------
# SORTERING (RETTET TIL AT MATCHE BLOKKEN)
# --------------------------

output_folder = os.path.join(folder, "sorted_tiles")
os.makedirs(output_folder, exist_ok=True)

# Matcher fx: 1_0_0.jpg, 74_4_3.jpeg osv.
tile_pattern = re.compile(r"^\d+_[0-4]_[0-4]\.(jpg|jpeg|png)$", re.IGNORECASE)

for filename in os.listdir(folder):
    if not tile_pattern.match(filename):
        continue

    path = os.path.join(folder, filename)
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
