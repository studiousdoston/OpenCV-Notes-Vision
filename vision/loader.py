import io
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

pillow_heif.register_heif_opener()

PDF_DPI = 200
MAX_PDF_SIDE = 2500


def load_pages(path: str | Path) -> list[np.ndarray]:
    return load_pages_from_bytes(Path(path).read_bytes())


def load_pages_from_bytes(data: bytes) -> list[np.ndarray]:
    if data.startswith(b"%PDF"):
        return _decode_pdf(data)
    return [_decode_with_pillow(data)]


def _decode_with_pillow(data: bytes) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as e:
        raise ValueError("File is not a supported image or PDF") from e

    with im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        rgb = np.array(im)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _decode_pdf(data: bytes) -> list[np.ndarray]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as e:
        raise ValueError("File is not a valid PDF") from e

    pages = []
    with doc:
        for page in doc:
            longest_pt = max(page.rect.width, page.rect.height)

            zoom = min(PDF_DPI / 72, MAX_PDF_SIDE / longest_pt)

            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)

            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return pages
