import argparse
import json
from pathlib import Path

from vision.pipeline import process_file

STATUS_MESSAGES = {
    "too_small": "image too small, use a higher-resolution photo",
    "blurry": "photo is blurry, please retake",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean photos of handwritten notes for AI flashcard generation."
    )
    parser.add_argument("files", nargs="+", type=Path, help="images or PDFs to process")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("output/cli"),
        help="folder for cleaned pages (default: output/cli)",
    )
    parser.add_argument("--json", action="store_true", help="also save report.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = []
    for path in args.files:
        try:
            results = process_file(path.read_bytes())
        except (ValueError, OSError) as e:
            print(f"✗ {path.name}: {e}")
            report.append({"file": path.name, "error": str(e)})
            continue

        for r in results:
            label = f"{path.name} p{r.page_number}"
            entry = {
                "file": path.name,
                "page": r.page_number,
                "status": r.status,
                "blur_score": round(r.blur_score, 1) if r.blur_score else None,
                "page_found": r.page_found,
                "size": f"{r.width}x{r.height}" if r.ok else None,
                "tokens": r.tokens,
            }

            if r.ok:
                out_path = (
                    args.out
                    / f"{path.stem}_{path.suffix[1:].lower()}_p{r.page_number}.jpg"
                )
                out_path.write_bytes(r.jpeg)
                entry["output"] = str(out_path)
                flat = "flattened" if r.page_found else "kept as-is"
                print(
                    f"✓ {label:28} {entry['size']:>10}  {r.tokens:5} tokens  ({flat})"
                )
            else:
                print(f"✗ {label:28} {STATUS_MESSAGES[r.status]}")
            report.append(entry)

    ok = [e for e in report if e.get("status") == "ok"]
    total_tokens = sum(e["tokens"] for e in ok)
    print("-" * 70)
    print(
        f"{len(ok)}/{len(report)} pages ready · {total_tokens} tokens total · saved to {args.out}"
    )
    # summary line: how many pages passed and the total AI cost in tokens

    if args.json:
        report_path = args.out / "report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Report saved to {report_path}")
    # optional machine-readable report (indent=2 makes it human-readable too)


if __name__ == "__main__":
    main()
# runs main() only when the file is executed directly,
# not when it's imported from somewhere else
