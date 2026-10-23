from pathlib import Path

from vision.compress import api_tokens, prepare_for_ai, resize_for_ai, visual_tokens
from vision.enhance import enhance
from vision.loader import load_pages
from vision.page import detect_and_flatten

SAMPLES = Path("samples")
OUT = Path("output/compressed")
OUT.mkdir(parents=True, exist_ok=True)

total_before_kb = total_after_kb = 0
total_before_tok = total_after_tok = 0
# running totals for a summary at the end

for file in sorted(SAMPLES.iterdir()):
    if file.name.startswith("."):
        continue
    try:
        pages = load_pages(file)
    except ValueError as e:
        print(f"SKIP {file.name}: {e}")
        continue

    file_kb = file.stat().st_size / 1024
    # original file size on disk, in KB

    for i, page in enumerate(pages, start=1):
        name = f"{file.stem}_{file.suffix[1:].lower()}_p{i}"
        flat, _ = detect_and_flatten(page)
        clean = enhance(flat)
        # pipeline so far: flatten → enhance

        data = prepare_for_ai(clean)
        (OUT / f"{name}.jpg").write_bytes(data)
        # resized + compressed bytes, saved so to inspect readability

        h, w = clean.shape[:2]
        rh, rw = resize_for_ai(clean).shape[:2]
        before_tok = api_tokens(w, h)
        after_tok = visual_tokens(rw, rh)
        after_kb = len(data) / 1024
        # tokens if sent full-size (high-res model) vs after our resize

        total_before_kb += file_kb
        total_after_kb += after_kb
        total_before_tok += before_tok
        total_after_tok += after_tok

        print(
            f"{name:22} {file_kb:7.0f} KB → {after_kb:5.0f} KB   "
            f"tokens {before_tok:5} → {after_tok:5}"
        )

print("-" * 70)
print(
    f"{'TOTAL':22} {total_before_kb:7.0f} KB → {total_after_kb:5.0f} KB   "
    f"tokens {total_before_tok:5} → {total_after_tok:5}"
)
print(
    f"Saved {100 * (1 - total_after_kb / total_before_kb):.0f}% file size, "
    f"{100 * (1 - total_after_tok / total_before_tok):.0f}% tokens"
)
