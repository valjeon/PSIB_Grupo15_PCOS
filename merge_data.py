from pathlib import Path
import shutil

SRC = Path("data/raw")
DST = Path("data/all")

for split in ("train", "test"):
    for cls_dir in (SRC / split).iterdir():
        if not cls_dir.is_dir():
            continue
        out = DST / cls_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for img in cls_dir.iterdir():
            shutil.copy2(img, out / f"{split}_{img.name}")