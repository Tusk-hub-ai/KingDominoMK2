from pathlib import Path
import cv2
SCALE = 5

SPACE = 32
ESC = 27
# 'e' you compare with ord('e')

path_input = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/SortedKing")
path_output = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/CrownsSize/takenfromcrowns")
lis = []
crown_size = 29
half = crown_size // 2


image_counter = [0]
current_image = [0]
current_path = [0]
last_saved = [None]


def mouseEvent(event, x , y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        orig_x,orig_y = x//SCALE, y//SCALE
        print(orig_x,orig_y)
        if orig_y-half < 0  or  orig_y + half +1 > current_image[0].shape[0] or orig_x- half < 0 or orig_x + half +1 > current_image[0].shape[1]:
            print("Too close to edges!!! Click rejected.")
            return
        image_sliced = current_image[0][orig_y-half : orig_y + half +1, orig_x- half : orig_x+ half +1 ]
        path = path_output / f"{current_path[0].stem}_image_{image_counter[0]}.png"
        output = cv2.imwrite(str(path), image_sliced)
        if not output:
            print("imwrite failed for:", path)
            return
        image_counter[0] += 1
        last_saved[0] = path
        print("saved:", path.name)
   

for file in path_input.iterdir():
    lis.append(file)
shouldquit = False
cv2.namedWindow("Picture")

cv2.setMouseCallback("Picture", mouseEvent)
for source_path in lis:
    image = cv2.imread(str(source_path))
    if image is None:
        print("Failed to load image:", source_path)
    else:
        current_image[0] = image
        current_path[0] = source_path
        image_counter[0] = 0
        last_saved[0] = None
        print(image.shape)
        image_scaled = cv2.resize(image, (image.shape[1]*SCALE, image.shape[0]*SCALE), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Picture",image_scaled)
        
        while True:
            key = cv2.waitKey(0)
            if key == SPACE:
                
                break
            elif key == ESC:
                shouldquit = True
                break
            elif key == ord("e"):
                if last_saved[0] is None:
                    print("nothing to undo")
                else:
                    path = last_saved[0]
                    last_saved[0].unlink()
                    image_counter[0] -= 1
                    last_saved[0] = None
                    print("undone:", path.name)
        if shouldquit:
            break
cv2.destroyAllWindows() 