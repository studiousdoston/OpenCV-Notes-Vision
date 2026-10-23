from dataclasses import dataclass

import numpy as np

from vision.compress import encode_jpeg, resize_for_ai, visual_tokens
from vision.enhance import enhance
from vision.loader import load_pages_from_bytes
from vision.page import detect_and_flatten
from vision.quality import BLUR_THRESHOLD, blur_score, is_too_small

# imports one main function from each step we built


@dataclass
class PageResult:
    page_number: int
    status: str
    blur_score: float | None = None
    page_found: bool = False
    width: int | None = None
    height: int | None = None
    tokens: int | None = None
    jpeg: bytes | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def process_page(img: np.ndarray, page_number: int) -> PageResult:
    if is_too_small(img):
        return PageResult(page_number, status="too_small")

    score = blur_score(img)
    if score < BLUR_THRESHOLD:
        return PageResult(page_number, status="blurry", blur_score=score)

    flat, found = detect_and_flatten(img)
    clean = enhance(flat)
    resized = resize_for_ai(clean)
    h, w = resized.shape[:2]
    # steps 4–6: flatten (if a page is found) → remove shadows → resize

    return PageResult(
        page_number,
        status="ok",
        blur_score=score,
        page_found=found,
        width=w,
        height=h,
        tokens=visual_tokens(w, h),
        jpeg=encode_jpeg(resized),
    )


def process_file(data: bytes) -> list[PageResult]:
    pages = load_pages_from_bytes(data)
    return [process_page(page, i) for i, page in enumerate(pages, start=1)]
