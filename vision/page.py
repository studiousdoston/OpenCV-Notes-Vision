import cv2
import numpy as np

WORK_HEIGHT = 800
BORDER_MARGIN_RATIO = 0.02
MIN_PAGE_AREA_RATIO = 0.25
MAX_PAGE_AREA_RATIO = 0.95


def page_mask(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = WORK_HEIGHT / h
    small = cv2.resize(img, (int(w * scale), WORK_HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask, scale


def find_page_corners(img: np.ndarray) -> np.ndarray | None:
    mask, scale = page_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    page = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(page) / mask.size
    if not (MIN_PAGE_AREA_RATIO <= area_ratio <= MAX_PAGE_AREA_RATIO):
        return None
    x, y, w, h = cv2.boundingRect(page)
    img_h, img_w = mask.shape
    margin = int(BORDER_MARGIN_RATIO * max(img_h, img_w))
    if x <= margin or y <= margin or x + w >= img_w - margin or y + h >= img_h - margin:
        return None
    hull = cv2.convexHull(page)
    perimeter = cv2.arcLength(hull, True)

    for eps in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32) / scale

    return None


def order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    # always the same order, so the warp knows which corner goes where


def flatten_page(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = ordered = order_corners(corners)
    # sorts the corners, and also unpacks them into four names

    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    # where the 4 corners should end up: a perfect upright rectangle

    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(img, matrix, (width, height))


def detect_and_flatten(img: np.ndarray) -> tuple[np.ndarray, bool]:
    corners = find_page_corners(img)
    if corners is None:
        return img, False
    return flatten_page(img, corners), True
    # main function the pipeline will call
    # returns (image, found): unchanged image + False if no page was detected


def draw_corners(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    debug = img.copy()
    thickness = max(3, img.shape[1] // 200)
    cv2.polylines(
        debug, [order_corners(corners).astype(np.int32)], True, (0, 255, 0), thickness
    )
    return debug
