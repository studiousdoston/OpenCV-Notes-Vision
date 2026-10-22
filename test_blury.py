from pathlib import Path

import cv2

from vision.loader import load_pages
from vision.quality import BLUR_THRESHOLD, blur_score, is_too_small

SAMPLES = Path("samples")

for file in sorted(SAMPLES.iterdir()):
    if file.name.startswith("."):
        continue
    try:
        pages = load_pages(file)
    except ValueError as e:
        print(f"SKIP {file.name}: {e}")
        continue

    for i, page in enumerate(pages, start=1):
        score = blur_score(page)
        if is_too_small(page):
            verdict = "SMALL"
        elif score < BLUR_THRESHOLD:
            verdict = "BLURRY"
        else:
            verdict = "OK"
        # size check first, then blur, the same order the real pipeline will use

        sigma = page.shape[1] / 500
        fake_blurry = cv2.GaussianBlur(page, (0, 0), sigmaX=sigma)
        fake_score = blur_score(fake_blurry)

        print(
            f"{file.name:22} p{i}  score={score:8.1f}  {verdict:6}"
            f"  |  blurred copy={fake_score:8.1f}"
        )
