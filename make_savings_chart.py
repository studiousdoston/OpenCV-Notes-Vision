from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vision.compress import api_tokens, prepare_for_ai, resize_for_ai, visual_tokens
from vision.enhance import enhance
from vision.loader import load_pages
from vision.page import detect_and_flatten

SAMPLES = Path("samples")
OUT = Path("docs/showcase")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for file in sorted(SAMPLES.iterdir()):
    if file.name.startswith("."):
        continue
    try:
        pages = load_pages(file)
    except ValueError:
        continue
    file_kb = file.stat().st_size / 1024

    for page in pages:
        clean = enhance(detect_and_flatten(page)[0])
        data = prepare_for_ai(clean)
        h, w = clean.shape[:2]
        rh, rw = resize_for_ai(clean).shape[:2]
        rows.append(
            (
                f"{file.stem}.{file.suffix[1:].lower()}",
                file_kb,
                len(data) / 1024,
                api_tokens(w, h),
                visual_tokens(rw, rh),
            )
        )

labels = [r[0] for r in rows]
x = np.arange(len(rows))
width = 0.38
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

charts = [
    (3, 4, "Claude visual tokens per page", "tokens"),
    (1, 2, "File size per page", "KB"),
]
for ax, (before_i, after_i, title, unit) in zip(axes, charts):
    before = [r[before_i] for r in rows]
    after = [r[after_i] for r in rows]
    # picks the right columns from each row for this chart

    ax.bar(x - width / 2, before, width, label="Before", color="#B4B2A9")
    ax.bar(x + width / 2, after, width, label="After", color="#1D9E75")
    # gray "before" bar and green "after" bar next to each other per page

    saved = 100 * (1 - sum(after) / sum(before))
    ax.set_title(f"{title}: {saved:.0f}% saved", fontsize=13, fontweight="bold")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(unit)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    # total % saved goes in the title; tilted labels so long names fit
    # removing the top/right border lines gives a cleaner look

fig.text(
    0.5,
    0.005,
    "'Before' tokens = what a high-resolution Claude model charges for the full-size image",
    ha="center",
    fontsize=9,
    color="gray",
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "step6_savings.png", dpi=150)
print(f"Saved {OUT / 'step6_savings.png'}")
