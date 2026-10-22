from pathlib import Path

import cv2

from vision.loader import load_pages

SAMPLES = Path("samples")
OUT = Path("output/loaded")
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
        h, w = page.shape[:2]
        out_path = OUT / f"{file.name}_p{i}.jpg"
        if not cv2.imwrite(str(out_path), page):
            raise RuntimeError(f"Failed to save {out_path}")

        print(f"{file.name} → page {i}: {w}x{h}px")
