from pathlib import Path

import cv2
import numpy as np

from vision.enhance import enhance
from vision.loader import load_pages
from vision.page import detect_and_flatten

SAMPLES = Path("samples")
OUT = Path("output/enhance")
OUT.mkdir(parents=True, exist_ok=True)

for file in sorted(SAMPLES.iterdir()):
    if file.name.startswith("."):
        continue
    try:
        pages = load_pages(file)
    except ValueError as e:
        print(f"SKIP {file.name}: {e}")
        continue

    for i, page in enumerate(pages, start=1):
        name = f"{file.stem}_{file.suffix[1:].lower()}_p{i}"
        flat, _ = detect_and_flatten(page)
        clean = enhance(flat)
        # runs the pipeline so far: flatten (if a page is found) → enhance

        gap = np.full((flat.shape[0], 20, 3), 255, dtype=np.uint8)
        side_by_side = np.hstack([flat, gap, clean])
        # before | white gap | after, in one image
        # works because enhance() keeps the exact same size

        path = OUT / f"{name}.jpg"
        if not cv2.imwrite(str(path), side_by_side):
            raise RuntimeError(f"Failed to save {path}")
        print(f"{name:22} → saved before/after")
