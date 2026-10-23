import cv2
import numpy as np

BG_SCALE = 0.25
BG_DILATE = 7
BG_MEDIAN = 21
INK_GAMMA = 1.4


def estimate_background(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    small = cv2.resize(
        channel, None, fx=BG_SCALE, fy=BG_SCALE, interpolation=cv2.INTER_AREA
    )
    small = cv2.dilate(small, np.ones((BG_DILATE, BG_DILATE), np.uint8))
    small = cv2.medianBlur(small, BG_MEDIAN)

    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def remove_shadows(img: np.ndarray) -> np.ndarray:
    result = []
    for channel in cv2.split(img):
        background = estimate_background(channel)
        result.append(cv2.divide(channel, background, scale=255))
    return cv2.merge(result)


def boost_ink(img: np.ndarray, gamma: float = INK_GAMMA) -> np.ndarray:
    table = np.array([255 * (i / 255) ** gamma for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def enhance(img: np.ndarray) -> np.ndarray:
    return boost_ink(remove_shadows(img))
