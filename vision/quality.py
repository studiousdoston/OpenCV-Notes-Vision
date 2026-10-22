import cv2
import numpy as np

ANALYSIS_WIDTH = 1000
BLUR_THRESHOLD = 250.0
# chosen from real samples: sharp photos scored 923–2902,
MIN_SHORT_SIDE = 600
# the shorter side of an image must be at least this many pixels


def blur_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w > ANALYSIS_WIDTH:
        new_h = int(h * ANALYSIS_WIDTH / w)
        gray = cv2.resize(gray, (ANALYSIS_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def is_blurry(img: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    return blur_score(img) < threshold


def is_too_small(img: np.ndarray, min_short_side: int = MIN_SHORT_SIDE) -> bool:
    h, w = img.shape[:2]
    return min(h, w) < min_short_side