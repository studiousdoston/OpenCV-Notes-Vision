import sys
from pathlib import Path

import cv2
import numpy as np

from vision.enhance import enhance
from vision.loader import load_pages
from vision.page import (
    detect_and_flatten,
    draw_corners,
    find_page_corners,
    flatten_page,
    page_mask,
)

PANEL_HEIGHT = 600
OUT = Path("docs/showcase")
OUT.mkdir(parents=True, exist_ok=True)
GAP = np.full((PANEL_HEIGHT, 12, 3), 255, dtype=np.uint8)


def fit_height(img: np.ndarray, height: int = PANEL_HEIGHT) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
    # resizes to a fixed height while keeping the aspect ratio


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out
    # black bar with white text at the top of a panel


def make_row(panels: list[np.ndarray]) -> np.ndarray:
    row = [panels[0]]
    for panel in panels[1:]:
        row += [GAP, panel]
    return np.hstack(row)
    # glues panels side by side with a gap between each


def save(path: Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise RuntimeError(f"Failed to save {path}")
    print(f"  saved {path}")
    # JPEG quality 85 keeps files small; prints each saved file


def showcase(src: Path) -> None:
    print(src.name)
    page = load_pages(src)[0]
    # first page only

    corners = find_page_corners(page)
    if corners is not None:
        mask, _ = page_mask(page)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        flat = flatten_page(page, corners)
        save(
            OUT / f"step4_{src.stem}.jpg",
            make_row(
                [
                    label(fit_height(page), "1. Original"),
                    label(fit_height(mask_bgr), "2. Page mask"),
                    label(fit_height(draw_corners(page, corners)), "3. Detected"),
                    label(fit_height(flat), "4. Flattened"),
                ]
            ),
        )
    else:
        flat, _ = detect_and_flatten(page)
        print("  no page found → skipping Step 4 (already cropped)")
    # Step 4 image only when a page was really detected
    # otherwise flat = the unchanged image, so Step 5 still works

    save(
        OUT / f"step5_{src.stem}.jpg",
        make_row(
            [
                label(fit_height(flat), "Before"),
                label(fit_height(enhance(flat)), "After: shadows removed"),
            ]
        ),
    )
    # Step 5 image for every file


for arg in sys.argv[1:]:
    showcase(Path(arg))
