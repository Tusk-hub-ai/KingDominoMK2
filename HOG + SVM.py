from features import extract_hog_features
from pathlib import Path
import cv2
import joblib

negatives_rand = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/CrownsSize/noCrowns")
negatives_hard = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/CrownsSize/takenfromcrowns")
positives = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/CrownsSize/rotateCrowns")


def extract_from_folder(folder, label,X,y,groups):
    for file in folder.iterdir():
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        features = extract_hog_features(image)
        X.append(features)
        y.append(label)
        board_id = file.name.split('_')[0]
        groups.append(board_id)
     


X = []
y = []
groups = []
extract_from_folder(negatives_rand, 0, X, y ,groups)
extract_from_folder(negatives_hard, 0, X, y,groups)
extract_from_folder(positives, 1, X, y,groups)

import numpy as np
X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]


model = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
print(f"train: {len(train_idx)} patches")
print(f"test:  {len(test_idx)} patches")
train_groups = set(groups[i] for i in train_idx)
test_groups = set(groups[i] for i in test_idx)
print(f"train groups: {len(train_groups)}")
print(f"test groups:  {len(test_groups)}")
print(f"overlap: {train_groups & test_groups}")
print(f"train class counts: {np.unique(y[train_idx], return_counts=True)}")
print(f"test class counts:  {np.unique(y[test_idx], return_counts=True)}")



svm_model = model.fit(X_train,y_train)
predictions = svm_model.predict(X_test)
results = classification_report(y_test,predictions)
cm = confusion_matrix(y_test, predictions)
print(results)
print(cm)

print(sorted(set(groups)))

joblib.dump(svm_model,Path(__file__).parent / "svm_model.pkl")