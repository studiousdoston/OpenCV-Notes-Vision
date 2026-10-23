from pathlib import Path

import cv2

from vision.loader import load_pages
from vision.page import draw_corners, find_page_corners, flatten_page, page_mask

SAMPLES = Path("samples")
OUT = Path("output/page")
OUT.mkdir(parents=True, exist_ok=True)


def save(path: Path, img) -> None:
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Failed to save {path}")


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
        mask, _ = page_mask(page)
        save(OUT / f"{name}_0_mask.jpg", mask)

        corners = find_page_corners(page)
        if corners is None:
            print(f"{name:22} → no page found, kept as-is")
            continue
        # cropped images land here, which is expected, not an error

        save(OUT / f"{name}_1_detected.jpg", draw_corners(page, corners))
        flat = flatten_page(page, corners)
        save(OUT / f"{name}_2_flat.jpg", flat)
        print(f"{name:22} → page found, flattened to {flat.shape[1]}x{flat.shape[0]}px")
        # saves 2 images per page: the green outline on the original,
        # and the flattened result, so to compare them side by side
