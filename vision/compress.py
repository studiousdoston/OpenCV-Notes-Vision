import math

import cv2
import numpy as np

MAX_LONG_EDGE = 1568
JPEG_QUALITY = 85
MAX_TOKENS = 1568
# MAX_TOKENS standard-tier token cap for Claude: the image must fit BOTH this
# and MAX_LONG_EDGE, otherwise the API downscales it again
# JPEG_QUALITY  → 85 keeps handwriting sharp; much lower creates blocky
# artifacts around text (Anthropic's docs warn about this)


def visual_tokens(w: float, h: float) -> int:
    return math.ceil(w / 28) * math.ceil(h / 28)
    # Claude's documented formula: one token per 28x28 pixel patch


def resize_for_ai(
    img: np.ndarray, max_edge: int = MAX_LONG_EDGE, max_tokens: int = MAX_TOKENS
) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_edge / max(h, w))

    while visual_tokens(w * scale, h * scale) > max_tokens:
        scale *= 0.98

    if scale >= 1.0:
        return img
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
    # imencode = like imwrite, but into memory instead of a file
    # returns raw bytes, ready to send to the Claude API or save to S3


def api_tokens(w: float, h: float, max_edge: int = 2576, max_tokens: int = 4784) -> int:
    scale = min(1.0, max_edge / max(w, h))
    w, h = w * scale, h * scale
    while visual_tokens(w, h) > max_tokens:
        w, h = w * 0.98, h * 0.98
    return visual_tokens(w, h)
    # estimates what a high-res model would charge for an image sent as-is:
    # it downscales to fit the 2576px edge, then shrinks until under 4784 tokens
    # used only for the before/after comparison


def prepare_for_ai(img: np.ndarray) -> bytes:
    return encode_jpeg(resize_for_ai(img))
    # main function the pipeline will call: resize, then compress to JPEG bytes
