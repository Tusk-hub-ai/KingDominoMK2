from pathlib import Path
import shutil

path_all = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/firstcut/tiles")
path_onlyKings = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/SortedKing")
path_noKings = Path(r"C:/exercices/miniproject/KingDominoMK2/pictures/SortedNonKings")
crown_names = {p.name for p in path_onlyKings.iterdir()}

for file in path_all.iterdir():
        print(f"checking: {file.name}")
        if file.name in crown_names:
            continue
        else:
            copy = shutil.copy(file, path_noKings / file.name)
print(f"crown_names has {len(crown_names)} entries")
print(f"first 3: {list(crown_names)[:3]}")