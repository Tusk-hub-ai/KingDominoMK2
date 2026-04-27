import joblib
from pathlib import Path
import cv2
from features import extract_hog_features


test_path = Path(__file__).parent / "test_data"


PATCH_SIZE = 29
STRIDE = 4
NMS_THRESHOLD = 0.4
DECISION_THRESHOLD = 0.6
TILE_SIZE = 100
BOARD_SIZE = 500
PAD = 10

def sweep_board(image, svm):
    height, width = image.shape[:2]
    candidates = []
    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            features = extract_hog_features(patch)
            score = svm.decision_function(features.reshape(1, -1))[0]
            if score > DECISION_THRESHOLD:
                candidates.append((x, y, score))
    return candidates


def compute_iou(box1, box2):
    x1a, y1a, x1b, y1b = box1
    x2a, y2a, x2b, y2b = box2
    overlap_x_start = max(x1a, x2a)
    overlap_x_end = min(x1b, x2b)
    overlap_width = max(0, overlap_x_end - overlap_x_start)
    overlap_y_start = max(y1a, y2a)
    overlap_y_end = min(y1b, y2b)
    overlap_height = max(0, overlap_y_end - overlap_y_start)
    intersection = overlap_width * overlap_height
    union = (PATCH_SIZE ** 2) * 2 - intersection
    iou = intersection / union
    return iou


def sort_candidates(candidates):
    runflag = True
    while runflag:
        hop = False
        for reagement in range(len(candidates) - 1):
            score_a = candidates[reagement][2]
            score_b = candidates[reagement + 1][2]
            if score_a < score_b:
                candidates[reagement], candidates[reagement + 1] = candidates[reagement + 1], candidates[reagement]
                hop = True
                continue
            else:
                continue
        if hop:
            continue
        runflag = False
    return candidates


def non_max_suppression(candidates, iou_threshold):
    kept = []
    for candidate in sort_candidates(candidates):
        x, y, score = candidate
        box_candy = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
        is_duplicate = False
        for item in kept:
            kx, ky, kscore = item
            box_kept = (kx, ky, kx + PATCH_SIZE, ky + PATCH_SIZE)
            iou = compute_iou(box_candy, box_kept)
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
    return kept


def detect_on_board(image, svm, board_id):

    image = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    candidates = sweep_board(image, svm)
    surviving = non_max_suppression(candidates, NMS_THRESHOLD)
    all_detections = []
    for survivor in surviving:
        x, y, score = survivor
        x = x - PAD
        y = y - PAD
    # now proceed with tile assignment using corrected coordinates
        centre_x = x + PATCH_SIZE // 2
        centre_y = y + PATCH_SIZE // 2
        tile_row = centre_y // TILE_SIZE
        tile_col = centre_x // TILE_SIZE
        # Pixel coords relative to the tile (keeps output format compatible)
        pixel_x = x - tile_col * TILE_SIZE
        pixel_y = y - tile_row * TILE_SIZE
        record = {
            "board_id": board_id,
            "tile_row": tile_row,
            "tile_col": tile_col,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "confidence": score,
        }
        all_detections.append(record)
    return all_detections


def visualize_detections(image, detections, output_path):
    for detection in detections:
        tile_row = detection["tile_row"]
        tile_col = detection["tile_col"]
        pixel_x = detection["pixel_x"]
        pixel_y = detection["pixel_y"]
        global_x = tile_col * TILE_SIZE + pixel_x
        global_y = tile_row * TILE_SIZE + pixel_y
        x1 = global_x
        y1 = global_y
        x2 = global_x + PATCH_SIZE
        y2 = global_y + PATCH_SIZE
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    cv2.imwrite(str(output_path), image)


def main(boards_path):
    results_dir = Path(__file__).parent / "Testresults"
    results_dir.mkdir(exist_ok=True)
    svm = joblib.load(Path(__file__).parent / "svm_model.pkl")
    for file in boards_path.iterdir():
        if file.suffix.lower() != '.jpg':
            continue
        board_id = file.stem
        output_path = results_dir / f"{board_id}_detected.jpg"
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Failed to load", file.name)
            continue
        detections = detect_on_board(image, svm, board_id)
        print(f"Board {board_id}: {len(detections)} crowns detected")
        colors = cv2.imread(str(file))
        visualize_detections(colors, detections, output_path)

    print("NMS", NMS_THRESHOLD)
    print("DES", DECISION_THRESHOLD)


if __name__ == "__main__":
    main(test_path)