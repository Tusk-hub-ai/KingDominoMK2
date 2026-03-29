import os
import cv2
import random

# ============================================================
# SETTINGS
# ============================================================
INPUT_FOLDER = "C:\Programmering\\aau\Semester 2\Mini Projekt\KINGDOMINO_SIUUUUUUUUUU1\SortedKing"
OUTPUT_FOLDER = r"C:\Programmering\aau\Semester 2\Mini Projekt\KINGDOMINO_SIUUUUUUUUUU1\kingsCut"

N_IMAGES = 60          # number of images to slice
RANDOM_SEED = 42

IMG_SIZE = 500         # images are 500x500
GRID_SIZE = 3          # 3x3 = 9 slices

# ============================================================
# SETUP
# ============================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get all image paths
image_paths = [
    os.path.join(INPUT_FOLDER, f)
    for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# Reproducible random selection
random.seed(RANDOM_SEED)
selected_images = random.sample(image_paths, min(N_IMAGES, len(image_paths)))

print(f"Selected {len(selected_images)} images")

# ============================================================
# SLICE FUNCTION
# ============================================================
def slice_image(img, base_name, output_folder):
    h, w = img.shape[:2]

    # Step size (not perfectly divisible → last tiles bigger)
    step_h = h // GRID_SIZE
    step_w = w // GRID_SIZE

    count = 0

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            y1 = i * step_h
            x1 = j * step_w

            # Make last row/col go to the edge
            y2 = (i + 1) * step_h if i < GRID_SIZE - 1 else h
            x2 = (j + 1) * step_w if j < GRID_SIZE - 1 else w

            tile = img[y1:y2, x1:x2]

            filename = f"{base_name}_tile_{count}.jpg"
            path = os.path.join(output_folder, filename)

            cv2.imwrite(path, tile)
            count += 1

# ============================================================
# PROCESS IMAGES
# ============================================================
for idx, path in enumerate(selected_images):
    img = cv2.imread(path)

    if img is None:
        print(f"Skipping unreadable image: {path}")
        continue

    # Optional: ensure size is exactly 500x500
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    base_name = f"img_{idx}"
    slice_image(img, base_name, OUTPUT_FOLDER)

print("Done slicing!")