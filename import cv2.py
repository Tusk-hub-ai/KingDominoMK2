import cv2
import os

# Get the folder where this script is located
folder = os.path.dirname(os.path.abspath(__file__))

# Create a list of all image files (1.jpg to 74.jpg) in that folder
pictures = [os.path.join(folder, f"{i}.jpg") for i in range(1, 75)]

image_size = 100

for picture_path in pictures:
    img = cv2.imread(picture_path)
    if img is None:
        print(f"Failed to load: {picture_path}")
        continue

    base_name = os.path.splitext(os.path.basename(picture_path))[0]  # e.g., '1' takes the number

    for row, height in enumerate(range(0, 500, image_size)):
        for col, length in enumerate(range(0, 500, image_size)):
            tile = img[height:height+image_size, length:length+image_size]

            # Save tile in the same folder as the script
            output_path = os.path.join(folder, f"{base_name}_{row}_{col}.jpg")
            cv2.imwrite(output_path, tile)