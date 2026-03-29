import cv2 as cv
import numpy as np
import os

# Main function containing the backbone of the program


def main():
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    image_path = r"C:\Users\rasmu\Desktop\project0\data\37.jpg"
    if not os.path.isfile(image_path):
        print("Image not found")
        return
    image = cv.imread(image_path)
    tiles = get_tiles(image)
    print(len(tiles))
    # Fra chatgpt
    terrain_counts = {}  # Dictionary to hold terrain counts
    for y, row in enumerate(tiles):     # gentager gennem rækker
        for x, tile in enumerate(row):  # gentager gennem kolonner
            print(f"Tile ({x}, {y}):")  # Udskriver koordinaterne for feltet
            terrain = get_terrain(tile)  # Bestemmer terræntypen for feltet
            print(terrain)  # Udskriver terræntypen
            # Opdaterer tælleren for terræntypen i dictionaryen
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
            print("=====")  # Separator for bedre læsbarhed

    print("\nTerrain counts:")
    for terrain, count in terrain_counts.items():
        print(f"{terrain}: {count}")

# Break a board into tiles


def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tiles[-1].append(image[y*100:(y+1)*100, x*100:(x+1)*100])
    return tiles

# Determine the type of terrain in a tile


def get_terrain(tile):
    hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    hue, saturation, value = np.median(hsv_tile, axis=(0, 1))
    print(f"H: {hue}, S: {saturation}, V: {value}")

    if 34 <= hue <= 48 and 169 <= saturation <= 248 and 73 <= value <= 164:
        return "Grassland"

    if 28 <= hue <= 77 and 68 <= saturation <= 239 and 25 <= value <= 128:
        return "Forest"

    if 78 <= hue <= 109 and 193 <= saturation <= 255 and 109 <= value <= 204:
        return "Lake"

    if 17 <= hue <= 26 and 23 <= saturation <= 181 and 68 <= value <= 144:
        return "Swamp"

    if 17 <= hue <= 30 and 34 <= saturation <= 159 and 24 <= value <= 95:
        return "Mine"

    if 20 <= hue <= 50 and 192 <= saturation <= 255 and 105 <= value <= 206:
        return "Field"

    if 18 <= hue <= 46 and 110 <= saturation <= 222 and 70 <= value <= 184:
        return "Empty Space"

    if 18 <= hue <= 39 and 54 <= saturation <= 181 and 64 <= value <= 150:
        return "Home"
    return "Unknown"


if __name__ == "__main__":
    main()
