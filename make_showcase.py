import sys
from pathlib import Path

import cv2
import numpy as np

from vision.loader import load_pages
from vision.page import draw_corners, find_page_corners, flatten_page, page_mask

PANEL_HEIGHT = 600
OUT = Path("docs/showcase")
OUT.mkdir(parents=True, exist_ok=True)


def fit_height(img: np.ndarray, height: int = PANEL_HEIGHT) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
    # resizes to a fixed height while keeping the aspect ratio


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out
    # draws a black bar with white text at the top of a panel
    # thickness -1 = a filled rectangle


def save(path: Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise RuntimeError(f"Failed to save {path}")


src = Path(sys.argv[1])
page = load_pages(src)[0]
corners = find_page_corners(page)
if corners is None:
    sys.exit(f"No page found in {src.name}, nothing to showcase")


mask, _ = page_mask(page)
mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


panels = [
    label(fit_height(page), "1. Original"),
    label(fit_height(mask_bgr), "2. Page mask"),
    label(fit_height(draw_corners(page, corners)), "3. Detected"),
    label(fit_height(flatten_page(page, corners)), "4. Flattened"),
]


gap = np.full((PANEL_HEIGHT, 12, 3), 255, dtype=np.uint8)
row = [panels[0]]
for panel in panels[1:]:
    row += [gap, panel]
save(OUT / f"step4_{src.stem}.jpg", np.hstack(row))


print(f"Saved {OUT / f'step4_{src.stem}.jpg'}")
