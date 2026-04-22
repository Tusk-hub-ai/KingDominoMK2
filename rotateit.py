import cv2
from pathlib import Path

input_path = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/CrownsSize/Crowns")
output_path = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/CrownsSize/rotateCrowns")
output_path.mkdir(parents=True, exist_ok=True)
rotations = [90,180,270]
for file in input_path.iterdir():
    file_name = file.stem
    if file.suffix != ".png":
        continue
    picture = cv2.imread(str(file))
    if picture is None:
        print("No paths for", file)
        continue

    path = output_path / f"{file_name}_rot0.png"
    success = cv2.imwrite(str(path), picture)
    if not success:
        print("imwrite failed for:", path)
    for rotate in rotations:
        if rotate == 90:
            rotation = cv2.ROTATE_90_CLOCKWISE
        elif rotate == 180:
            rotation = cv2.ROTATE_180
        else:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        path = output_path / f"{file_name}_rot{rotate}.png"
        rotated = cv2.rotate(picture, rotation)
        success = cv2.imwrite(str(path), rotated)
        if not success:
            print("imwrite failed for:", path)
