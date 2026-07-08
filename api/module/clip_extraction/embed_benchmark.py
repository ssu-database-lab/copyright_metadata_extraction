"""
Embedding / retrieval benchmark — measures CLIP-family models as EMBEDDING
models for the similarity + duplicate-detection (FAISS) role, NOT as zero-shot
classifiers. This is the job CLIP/SigLIP are actually good at, and what the
proposal's 중복검출 (89→92%) KPI needs.

Method (works even with a small image set):
  1. Embed each original image.
  2. Synthesize near-duplicates of each (rescale, JPEG recompress, crop,
     brightness shift) — simulating re-uploads / light edits that 중복검출
     must catch.
  3. For each perturbed query, retrieve the nearest original by cosine
     similarity. Rank-1 hit on the TRUE source = correct duplicate detection.
  4. Report per model: duplicate rank-1 accuracy, true-pair vs distractor
     similarity margin (separation), embedding dim, embed latency.

Usage:
    python -m api.module.clip_extraction.embed_benchmark
    python -m api.module.clip_extraction.embed_benchmark --models siglip2-so400m-patch14,koclip-bingsu-vit-l14
    python -m api.module.clip_extraction.embed_benchmark --images /path/to/imgs

Outputs:
    reports/embed_benchmark_{TIMESTAMP}.{json,md}
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from api.module.clip_extraction.models import REGISTRY  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = THIS_DIR / "test_data" / "sample_works"
REPORTS_DIR = THIS_DIR / "reports"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# Models that don't expose embeddings (generative VLMs) are skipped automatically.
DEFAULT_MODELS = [
    "siglip2-so400m-patch14",
    "koclip-bingsu-vit-l14",
    "multilingual-clip-vit-b32",
    "openai-clip-vit-l14",
    "jina-clip-v2",
]


# ---------------------------------------------------------- perturbations ---
def perturb(img: Image.Image) -> dict[str, Image.Image]:
    """Make near-duplicate variants simulating re-uploads / light edits."""
    w, h = img.size
    out: dict[str, Image.Image] = {}
    # 1. downscale to 60% then back up (resampling artifacts)
    small = img.resize((max(1, int(w * 0.6)), max(1, int(h * 0.6))), Image.LANCZOS)
    out["rescale60"] = small.resize((w, h), Image.LANCZOS)
    # 2. heavy JPEG recompression
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    buf.seek(0)
    out["jpeg40"] = Image.open(buf).convert("RGB")
    # 3. center crop 85% then resize back (reframing)
    cw, ch = int(w * 0.85), int(h * 0.85)
    left, top = (w - cw) // 2, (h - ch) // 2
    out["crop85"] = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)
    # 4. brightness +15%
    out["bright115"] = ImageEnhance.Brightness(img).enhance(1.15)
    return out


def discover_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


# ----------------------------------------------------------------- runner ---
def run_model(model_key: str, originals: list[Image.Image],
              names: list[str]) -> dict[str, Any]:
    cls = REGISTRY[model_key]
    inst = cls()
    res: dict[str, Any] = {"model_key": model_key, "info": None, "error": None}
    try:
        inst.load()
        res["info"] = inst.info()

        # Embed originals
        t0 = time.perf_counter()
        orig_emb = np.stack([inst.encode_image(im) for im in originals])  # (N,d), normalized
        embed_time = time.perf_counter() - t0
        n, d = orig_emb.shape

        # Perturbed queries → retrieve nearest original
        n_queries = 0
        n_rank1 = 0
        true_sims, distractor_sims = [], []
        per_variant: dict[str, dict[str, int]] = {}
        for i, im in enumerate(originals):
            for vname, vimg in perturb(im).items():
                q = inst.encode_image(vimg)            # (d,), normalized
                sims = orig_emb @ q                     # cosine (both normalized)
                pred = int(np.argmax(sims))
                hit = (pred == i)
                n_queries += 1
                n_rank1 += int(hit)
                true_sims.append(float(sims[i]))
                # mean similarity to all other originals (distractors)
                mask = np.ones(n, dtype=bool); mask[i] = False
                distractor_sims.append(float(sims[mask].mean()) if n > 1 else 0.0)
                pv = per_variant.setdefault(vname, {"q": 0, "hit": 0})
                pv["q"] += 1; pv["hit"] += int(hit)

        # Most-similar original pairs (qualitative — would they false-positive?)
        sim_mat = orig_emb @ orig_emb.T
        np.fill_diagonal(sim_mat, -1)
        top_pairs = []
        for _ in range(min(3, n * (n - 1) // 2 if n > 1 else 0)):
            idx = int(np.argmax(sim_mat))
            r, c = divmod(idx, n)
            top_pairs.append((names[r], names[c], round(float(sim_mat[r, c]), 3)))
            sim_mat[r, c] = sim_mat[c, r] = -1

        res["metrics"] = {
            "dim": d,
            "dup_rank1_accuracy": round(n_rank1 / n_queries, 4) if n_queries else None,
            "mean_true_pair_sim": round(float(np.mean(true_sims)), 4) if true_sims else None,
            "mean_distractor_sim": round(float(np.mean(distractor_sims)), 4) if distractor_sims else None,
            "separation_margin": round(float(np.mean(true_sims) - np.mean(distractor_sims)), 4)
                                  if true_sims else None,
            "embed_time_per_img_s": round(embed_time / n, 3) if n else None,
            "per_variant_rank1": {k: round(v["hit"] / v["q"], 3) for k, v in per_variant.items()},
            "top_similar_pairs": top_pairs,
        }
        m = res["metrics"]
        print(f"  ✓ {model_key}: dup_rank1={m['dup_rank1_accuracy']} "
              f"margin={m['separation_margin']} dim={d} {m['embed_time_per_img_s']}s/img")
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        print(f"  ✗ {model_key} FAILED: {e}")
    finally:
        try:
            inst.unload()
        except Exception:  # noqa: BLE001
            pass
        del inst; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res


def write_md(report: dict[str, Any], out: Path) -> None:
    L = [f"# Embedding / Duplicate-Detection Benchmark — {report['timestamp']}", "",
         f"- Images: {report['n_images']} (× {report['n_variants']} perturbations each = "
         f"{report['n_images'] * report['n_variants']} duplicate queries)",
         f"- Device: `{report['device']}`",
         "- Metric: retrieve the source original for each near-duplicate query (rank-1).",
         "  Higher dup_rank1 + larger margin = better for FAISS 중복검출.", ""]
    L += ["## Results", "",
          "| Model | License | Dim | Dup rank-1 | True sim | Distractor sim | Margin | Embed/img |",
          "|---|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        if r["error"]:
            L.append(f"| `{r['model_key']}` | — | — | **FAILED** | — | — | — | — |")
            continue
        i, m = r["info"], r["metrics"]
        acc = f"{m['dup_rank1_accuracy']*100:.1f}%" if m["dup_rank1_accuracy"] is not None else "n/a"
        L.append(f"| `{r['model_key']}` | {i['license']} | {m['dim']} | **{acc}** | "
                 f"{m['mean_true_pair_sim']} | {m['mean_distractor_sim']} | "
                 f"{m['separation_margin']} | {m['embed_time_per_img_s']}s |")
    L += ["", "## Per-perturbation rank-1 (robustness)", ""]
    for r in report["results"]:
        if r["error"]:
            continue
        L.append(f"- `{r['model_key']}`: {r['metrics']['per_variant_rank1']}")
    L += ["", "## Most-similar original pairs (false-positive risk)", ""]
    for r in report["results"]:
        if r["error"]:
            continue
        L.append(f"- `{r['model_key']}`: {r['metrics']['top_similar_pairs']}")
    out.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Embedding/retrieval benchmark for FAISS role")
    ap.add_argument("--models", default="", help="Comma-separated registry keys (default: embedding set)")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    args = ap.parse_args()

    image_dir = Path(args.images)
    paths = discover_images(image_dir)
    if not paths:
        print(f"No images in {image_dir}")
        return 1
    originals = [Image.open(p).convert("RGB") for p in paths]
    names = [p.name for p in paths]
    print(f"Loaded {len(originals)} images from {image_dir}")

    keys = [k.strip() for k in args.models.split(",") if k.strip()] or DEFAULT_MODELS
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"Unknown keys: {unknown}\nAvailable: {list(REGISTRY)}")
        return 2

    print(f"Models: {keys}\n")
    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_dir": str(image_dir),
        "n_images": len(originals),
        "n_variants": 4,
        "results": [],
    }
    for k in keys:
        print(f"=== {k} ===")
        report["results"].append(run_model(k, originals, names))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (REPORTS_DIR / f"embed_benchmark_{ts}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(report, REPORTS_DIR / f"embed_benchmark_{ts}.md")
    print(f"\nReports written:\n  {REPORTS_DIR}/embed_benchmark_{ts}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
