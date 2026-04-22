from pathlib import Path
import cv2
from random import randint


path_input = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/SortedNonKings")
path_output = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/CrownsSize/noCrowns")
crown_size = 29
half = crown_size // 2
path_output.mkdir(parents=True, exist_ok=True) 



for file in path_input.iterdir():
    image = cv2.imread(str(file))
    if image is None:
        print("failed:", file)
        continue
    for i in range(20):
        x = randint(half,100-half-1)
        y = randint(half,100-half-1)
        patch = image[y-half : y+half+1, x-half : x+half+1]
        dest = path_output / f"{file.stem}_neg_{i}.png" 
        ok = cv2.imwrite(str(dest), patch)
        if not ok:
            print("imwrite failed:", dest)

