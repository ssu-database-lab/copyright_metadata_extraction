"""
Benchmark runner — load each model in api/module/clip_extraction/models/__init__.py
REGISTRY, run zero-shot classification + (optional) image-text similarity on
the test image set, and write a comparison report.

Usage:
    # Run all registered models on the default test set
    python -m api.module.clip_extraction.benchmark

    # Run a subset (comma-separated registry keys)
    python -m api.module.clip_extraction.benchmark --models openai-clip-vit-l14,siglip2-so400m-patch14

    # Point at a custom image directory
    python -m api.module.clip_extraction.benchmark --images /path/to/images/

    # Set ground-truth labels by filename prefix (see test_data/README.md)
    python -m api.module.clip_extraction.benchmark --labels-from-filename

Outputs:
    api/module/clip_extraction/reports/benchmark_{TIMESTAMP}.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from api.module.clip_extraction.labels import label_set  # noqa: E402
from api.module.clip_extraction.models import REGISTRY    # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = THIS_DIR / "test_data" / "sample_works"
REPORTS_DIR = THIS_DIR / "reports"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


# ---------------------------------------------------------------- helpers ---
def discover_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def gt_label_from_filename(path: Path) -> str | None:
    """
    Parse ground-truth label from filename prefix.
    Convention: {label}__{anything}.{ext}
    Example: 사진저작물__sunset_beach.jpg → "사진저작물"
    """
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    return None


def memory_snapshot() -> dict[str, float]:
    """Return current process + GPU memory in MB."""
    snap: dict[str, float] = {}
    try:
        import psutil
        proc = psutil.Process()
        snap["rss_mb"] = proc.memory_info().rss / 1024 / 1024
    except ImportError:
        snap["rss_mb"] = -1.0
    if torch.cuda.is_available():
        snap["cuda_alloc_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        snap["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    return snap


# ---------------------------------------------------------------- runner ----
def run_model_on_images(
    model_key: str,
    images: list[tuple[Path, Image.Image, str | None]],
    candidate_labels: list[str],
    display_labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run one model on the full image set. Returns per-image results + summary.

    `candidate_labels` are what's passed to the model (may include a hypothesis
    template wrap). `display_labels` are the bare labels used for accuracy
    scoring and the report — if None, they default to candidate_labels.
    """
    cls = REGISTRY[model_key]
    instance = cls()
    if display_labels is None:
        display_labels = candidate_labels
    # templated → bare label, used to map model output back for scoring/display
    label_map = dict(zip(candidate_labels, display_labels))

    result: dict[str, Any] = {
        "model_key": model_key,
        "info": None,
        "memory_before_load": memory_snapshot(),
        "memory_after_load": None,
        "memory_after_inference": None,
        "per_image": [],
        "summary": {},
        "error": None,
    }

    try:
        print(f"  Loading {model_key} ({cls.__name__}) ...")
        instance.load()
        result["info"] = instance.info()
        result["memory_after_load"] = memory_snapshot()

        n_correct = 0
        n_with_gt = 0
        total_latency = 0.0

        for path, image, gt in images:
            t0 = time.perf_counter()
            try:
                raw_scores = instance.classify(image, candidate_labels)
                # Re-key under the bare display labels for scoring + reporting
                scores = {label_map.get(k, k): v for k, v in raw_scores.items()}
                latency = time.perf_counter() - t0
                top = sorted(scores.items(), key=lambda kv: -kv[1])
                top1 = top[0][0]
                correct = (gt is not None) and (top1 == gt)
                if gt is not None:
                    n_with_gt += 1
                    if correct:
                        n_correct += 1
                total_latency += latency
                result["per_image"].append({
                    "image": path.name,
                    "ground_truth": gt,
                    "top1": top1,
                    "top1_prob": round(top[0][1], 4),
                    "top3": [(lbl, round(p, 4)) for lbl, p in top[:3]],
                    "correct": correct if gt is not None else None,
                    "latency_s": round(latency, 3),
                })
            except Exception as e:  # noqa: BLE001
                result["per_image"].append({
                    "image": path.name,
                    "error": f"{type(e).__name__}: {e}",
                })

        result["memory_after_inference"] = memory_snapshot()
        n_images = len([r for r in result["per_image"] if "error" not in r])
        result["summary"] = {
            "images_processed": n_images,
            "images_with_ground_truth": n_with_gt,
            "top1_accuracy": round(n_correct / n_with_gt, 4) if n_with_gt > 0 else None,
            "avg_latency_s": round(total_latency / n_images, 3) if n_images > 0 else None,
            "total_time_s": round(total_latency, 2),
        }
        print(
            f"  ✓ {model_key}: "
            f"acc={result['summary']['top1_accuracy']} "
            f"avg={result['summary']['avg_latency_s']}s/img"
        )
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"  ✗ {model_key} FAILED: {e}")
    finally:
        try:
            instance.unload()
        except Exception:  # noqa: BLE001
            pass
        del instance
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------- report ----
def write_markdown_report(report: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# CLIP Benchmark Report — {report['timestamp']}")
    lines.append("")
    lines.append(f"- Label set: **{report['label_set']}** ({len(report['candidate_labels'])} labels)")
    lines.append(f"- Test images: {report['n_images']}")
    lines.append(f"- Device: `{report['device']}`")
    if report.get("hypothesis_template"):
        lines.append(f"- Hypothesis template: `{report['hypothesis_template']}`")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Korean | License | Params | Top-1 Acc | Avg Latency | Load Time | RSS Δ (MB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in report["results"]:
        if r["error"]:
            lines.append(f"| `{r['model_key']}` | — | — | — | **FAILED** | — | — | — |")
            continue
        info = r["info"]
        summ = r["summary"]
        rss_delta = ""
        try:
            rss_delta = f"{r['memory_after_load']['rss_mb'] - r['memory_before_load']['rss_mb']:.0f}"
        except (KeyError, TypeError):
            pass
        acc = summ.get("top1_accuracy")
        acc_str = f"{acc*100:.1f}%" if acc is not None else "n/a"
        lines.append(
            f"| `{r['model_key']}` | {info['korean_support']} | {info['license']} | "
            f"{info['params']} | {acc_str} | {summ['avg_latency_s']}s | "
            f"{info['load_time_s']}s | {rss_delta} |"
        )

    # Per-image breakdown for each model
    lines.append("")
    lines.append("## Per-image predictions")
    lines.append("")
    for r in report["results"]:
        lines.append(f"### `{r['model_key']}`")
        if r["error"]:
            lines.append(f"FAILED: {r['error']}")
            continue
        lines.append("")
        lines.append("| Image | GT | Top-1 | Prob | Correct |")
        lines.append("|---|---|---|---|---|")
        for row in r["per_image"]:
            if "error" in row:
                lines.append(f"| {row['image']} | — | ERROR: {row['error']} | — | — |")
                continue
            mark = "✓" if row.get("correct") else ("✗" if row["ground_truth"] else "—")
            lines.append(
                f"| {row['image']} | {row['ground_truth'] or '—'} | "
                f"{row['top1']} | {row['top1_prob']} | {mark} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------- main -----
def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark vision-language models")
    parser.add_argument("--models", default="", help="Comma-separated registry keys (default: all)")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES), help="Image directory")
    parser.add_argument("--label-set", default="work_type",
                        choices=["work_type", "subject", "license"])
    parser.add_argument("--language", default="ko", choices=["ko", "en"])
    parser.add_argument("--labels-from-filename", action="store_true",
                        help="Parse ground-truth label from filename prefix ({label}__name.jpg)")
    parser.add_argument("--hypothesis-template", default="",
                        help="Wrap each label, e.g. '이 이미지의 저작물 유형은 {label}이다'. "
                             "Use {label} as the placeholder. Empty = use bare labels.")
    args = parser.parse_args()

    image_dir = Path(args.images)
    image_paths = discover_images(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}")
        print(f"Drop test images into {image_dir}/ and re-run.")
        print("Hint: name them like '사진저작물__beach.jpg' to get accuracy scoring.")
        return 1

    # Load images once
    print(f"Loading {len(image_paths)} images from {image_dir}")
    images: list[tuple[Path, Image.Image, str | None]] = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            gt = gt_label_from_filename(p) if args.labels_from_filename else None
            images.append((p, img, gt))
            print(f"  - {p.name} ({img.size[0]}x{img.size[1]}) gt={gt}")
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipped {p.name}: {e}")

    # Choose models
    if args.models:
        keys = [k.strip() for k in args.models.split(",")]
        unknown = [k for k in keys if k not in REGISTRY]
        if unknown:
            print(f"Unknown model keys: {unknown}")
            print(f"Available: {list(REGISTRY)}")
            return 2
    else:
        keys = list(REGISTRY)

    display_labels = label_set(args.label_set, args.language)
    if args.hypothesis_template:
        if "{label}" not in args.hypothesis_template:
            print(f"--hypothesis-template must contain '{{label}}' placeholder")
            return 3
        candidate_labels = [args.hypothesis_template.format(label=lbl) for lbl in display_labels]
        print(f"Hypothesis template: {args.hypothesis_template!r}")
        print(f"Templated labels: {candidate_labels}")
    else:
        candidate_labels = display_labels
    print(f"Label set: {args.label_set} ({args.language}) — {display_labels}")
    print(f"Models to run: {keys}")
    print()

    # Run
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "label_set": args.label_set,
        "language": args.language,
        "candidate_labels": display_labels,
        "templated_labels": candidate_labels if args.hypothesis_template else None,
        "hypothesis_template": args.hypothesis_template or None,
        "image_dir": str(image_dir),
        "n_images": len(images),
        "results": [],
    }
    for key in keys:
        print(f"=== {key} ===")
        report["results"].append(
            run_model_on_images(key, images, candidate_labels, display_labels)
        )

    # Write reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"benchmark_{ts}.json"
    md_path = REPORTS_DIR / f"benchmark_{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report, md_path)
    print()
    print(f"Reports written:\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
