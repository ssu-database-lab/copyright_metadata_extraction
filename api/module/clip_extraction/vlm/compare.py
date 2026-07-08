"""
VLM comparison runner — Gemma 4 vs Qwen3-VL on the same images.

Runs both models with the identical extraction prompt over the test image
set, then writes a side-by-side report (description / work_type / keywords /
latency) so you can decide which VLM to use for attribute extraction.

Usage:
    # Compare both (default)
    python -m api.module.clip_extraction.vlm.compare

    # Gemma server on a different host (LAN IP or SSH tunnel target)
    python -m api.module.clip_extraction.vlm.compare --gemma-url http://192.168.0.42:8001/v1

    # Only one backend
    python -m api.module.clip_extraction.vlm.compare --models qwen
    python -m api.module.clip_extraction.vlm.compare --models gemma

    # Custom image dir
    python -m api.module.clip_extraction.vlm.compare --images /path/to/images

Outputs:
    api/module/clip_extraction/reports/vlm_compare_{TIMESTAMP}.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Make project root + api/ importable
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

import module.env_loader  # noqa: E402,F401 — loads .env (DASHSCOPE_API_KEY)
from api.module.clip_extraction.vlm.client import VLMClient, VLMResult  # noqa: E402
from api.module.clip_extraction.vlm.prompts import SYSTEM_PROMPT, USER_PROMPT  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = THIS_DIR.parent / "test_data" / "sample_works"
REPORTS_DIR = THIS_DIR.parent / "reports"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Backend definitions ------------------------------------------------------
DASHSCOPE_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_GEMMA_URL = "http://127.0.0.1:8001/v1"
GEMMA_MODEL = "google/gemma-4-31B-it"
QWEN_MODEL = "qwen3-vl-235b-a22b-instruct"


def discover_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def build_clients(which: list[str], gemma_url: str) -> list[VLMClient]:
    clients: list[VLMClient] = []
    if "openrouter" in which:
        # Reuse the production factory so this client gets the *authenticated*
        # ping (checks /api/v1/key). OpenRouter's /models is public, so a plain
        # models.list() ping would wrongly pass without a key and then 401 on
        # every image — the exact trap extractor.make_openrouter_backend fixes.
        from api.module.clip_extraction.vlm.extractor import make_openrouter_backend
        clients.append(make_openrouter_backend())
    if "gemma" in which:
        clients.append(VLMClient(
            model_label="Gemma 4 31B (vLLM)",
            base_url=gemma_url,
            model=GEMMA_MODEL,
            api_key="not-needed",
            image_first=True,  # Gemma prefers image-first
        ))
    if "qwen" in which:
        clients.append(VLMClient(
            model_label="Qwen3-VL-235B (DashScope)",
            base_url=DASHSCOPE_BASE,
            model=QWEN_MODEL,
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            image_first=False,  # matches existing OCR pipeline ordering
        ))
    return clients


def write_markdown(report: dict[str, Any], out_path: Path) -> None:
    L: list[str] = []
    L.append(f"# VLM Comparison — {report['timestamp']}")
    L.append("")
    L.append(f"- Models: {', '.join(report['models'])}")
    L.append(f"- Images: {report['n_images']}")
    L.append("")

    # Latency / parse-success summary
    L.append("## Summary")
    L.append("")
    L.append("| Model | Images OK | JSON parse OK | Avg latency | Avg completion tokens |")
    L.append("|---|---|---|---|---|")
    for label, agg in report["summary"].items():
        L.append(
            f"| {label} | {agg['ok']}/{report['n_images']} | "
            f"{agg['parsed']}/{report['n_images']} | "
            f"{agg['avg_latency']}s | {agg['avg_completion_tokens']} |"
        )
    L.append("")

    # Per-image side-by-side
    L.append("## Per-image extraction")
    for img in report["images"]:
        L.append("")
        L.append(f"### {img}")
        L.append("")
        for label in report["models"]:
            rec = report["by_image"][img].get(label)
            if not rec:
                continue
            if not rec["ok"]:
                L.append(f"**{label}** — ERROR: {rec['error']}")
                L.append("")
                continue
            p = rec.get("parsed") or {}
            L.append(f"**{label}** ({rec['latency_s']}s):")
            if p:
                L.append(f"- work_type: **{p.get('work_type', '—')}** "
                         f"({p.get('work_type_reason', '')})")
                L.append(f"- description: {p.get('description', '—')}")
                kws = p.get("keywords") or []
                L.append(f"- keywords: {', '.join(kws) if isinstance(kws, list) else kws}")
                subs = p.get("main_subjects") or []
                L.append(f"- subjects: {', '.join(subs) if isinstance(subs, list) else subs}")
                cols = p.get("dominant_colors") or []
                L.append(f"- colors: {', '.join(cols) if isinstance(cols, list) else cols}")
                if p.get("text_in_image"):
                    L.append(f"- text_in_image: {p.get('text_in_image')}")
            else:
                # JSON parse failed — show raw text so it's not lost
                raw = rec.get("raw_text", "")[:500]
                L.append(f"- (JSON parse failed) raw: {raw}")
            L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare VLMs for attribute extraction")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--models", default="gemma,qwen",
                        help="Comma-separated: gemma, qwen, openrouter (default: gemma,qwen)")
    parser.add_argument("--gemma-url", default=os.getenv("GEMMA_URL", DEFAULT_GEMMA_URL),
                        help="Gemma vLLM base URL (env GEMMA_URL also works)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images (0=all)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-lang", default="ko", choices=["ko", "en"],
                        help="Instruction language: 'ko' (default) or 'en' (values still Korean)")
    args = parser.parse_args()
    from api.module.clip_extraction.vlm.prompts import get_prompts
    sys_prompt, usr_prompt = get_prompts(args.prompt_lang)
    print(f"Prompt language: {args.prompt_lang}")

    which = [m.strip() for m in args.models.split(",") if m.strip()]
    image_dir = Path(args.images)
    images = discover_images(image_dir)
    if not images:
        print(f"No images in {image_dir}")
        return 1
    if args.limit and len(images) > args.limit:
        # evenly-spaced sample across the (idx-sorted) set for representativeness
        step = len(images) / args.limit
        images = [images[int(i * step)] for i in range(args.limit)]

    clients = build_clients(which, args.gemma_url)
    if not clients:
        print(f"No valid backends in --models={args.models} "
              f"(use gemma, openrouter, and/or qwen)")
        return 2

    # Reachability check up front — fail fast with a clear message
    print("Checking backends...")
    live_clients = []
    for c in clients:
        ok, detail = c.ping()
        status = "✓" if ok else "✗"
        print(f"  {status} {c.model_label}: {detail}")
        if ok:
            live_clients.append(c)
        elif c.model_label == "Gemma 4 31B (vLLM)":
            print(f"    → local Gemma unreachable at {args.gemma_url}. "
                  f"Pass --gemma-url with the server's LAN IP, or open an SSH tunnel:")
            print(f"      ssh -L 8001:127.0.0.1:8001 <gemma-host>")
        elif "OpenRouter" in c.model_label:
            print(f"    → OpenRouter unreachable: check OPENROUTER_API_KEY in .env "
                  f"and network connectivity.")
    if not live_clients:
        print("No reachable backends — aborting.")
        return 3

    print(f"\nProcessing {len(images)} images with {len(live_clients)} model(s)...\n")

    by_image: dict[str, dict[str, Any]] = {}
    agg: dict[str, dict[str, Any]] = {
        c.model_label: {"ok": 0, "parsed": 0, "lat": [], "ctoks": []}
        for c in live_clients
    }

    for img_path in images:
        by_image[img_path.name] = {}
        print(f"=== {img_path.name} ===")
        for c in live_clients:
            res: VLMResult = c.extract(
                img_path, sys_prompt, usr_prompt,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            by_image[img_path.name][c.model_label] = res.__dict__
            a = agg[c.model_label]
            if res.ok:
                a["ok"] += 1
                a["lat"].append(res.latency_s)
                if res.usage.get("completion_tokens"):
                    a["ctoks"].append(res.usage["completion_tokens"])
                if res.parse_ok:
                    a["parsed"] += 1
                    wt = (res.parsed or {}).get("work_type", "?")
                    print(f"  ✓ {c.model_label}: work_type={wt} ({res.latency_s}s)")
                else:
                    print(f"  ⚠ {c.model_label}: ran but JSON parse failed ({res.latency_s}s)")
            else:
                print(f"  ✗ {c.model_label}: {res.error}")

    # Summaries
    summary = {}
    for label, a in agg.items():
        n_lat = len(a["lat"]) or 1
        n_tok = len(a["ctoks"]) or 1
        summary[label] = {
            "ok": a["ok"],
            "parsed": a["parsed"],
            "avg_latency": round(sum(a["lat"]) / n_lat, 2) if a["lat"] else None,
            "avg_completion_tokens": round(sum(a["ctoks"]) / n_tok) if a["ctoks"] else None,
        }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt_lang": args.prompt_lang,
        "models": [c.model_label for c in live_clients],
        "n_images": len(images),
        "image_dir": str(image_dir),
        "images": [p.name for p in images],
        "summary": summary,
        "by_image": by_image,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"vlm_compare_{ts}.json"
    md_path = REPORTS_DIR / f"vlm_compare_{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"\nReports written:\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
