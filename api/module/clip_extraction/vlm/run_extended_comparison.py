#!/usr/bin/env python3
"""
Pre-staged Gemma-vs-Qwen extended comparison on the curated diverse KOGL set.

Fires the moment Gemma is reachable again (via Tailscale). It:
  1. Resolves the Gemma URL (--gemma-url arg, else $GEMMA_URL env).
  2. Pings both backends and reports which are live.
  3. Runs the verified comparison (vlm.compare) on the curated 15-image set
     (api/module/clip_extraction/test_data/vlm_compare_set/), both models.
  4. Joins the resulting report with _labels.csv and prints a Gemma-vs-Qwen-vs-KOGL
     agreement summary.

Usage (once Tailscale is up and you have the Gemma host's 100.x address):
    cd /home/mbmk92/copyright/copyright_metadata_extraction && source venv/bin/activate
    python -m api.module.clip_extraction.vlm.run_extended_comparison \
        --gemma-url http://<gemma-tailnet-ip>:8001/v1

    # or export GEMMA_URL first, then run with no args.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # project root
SET_DIR = ROOT / "api/module/clip_extraction/test_data/vlm_compare_set"
REPORTS = ROOT / "api/module/clip_extraction/reports"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run pre-staged Gemma-vs-Qwen extended comparison")
    ap.add_argument("--gemma-url", default=os.getenv("GEMMA_URL", ""),
                    help="Gemma vLLM base URL, e.g. http://100.x.x.x:8001/v1 (or set $GEMMA_URL)")
    ap.add_argument("--images", default=str(SET_DIR), help="Image dir (default: curated set)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    if not args.gemma_url:
        print("No Gemma URL given. Pass --gemma-url http://<gemma-tailnet-ip>:8001/v1 "
              "(or export GEMMA_URL). Aborting so we don't run a Qwen-only pass by accident.")
        return 2

    img_dir = Path(args.images)
    n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    if n_imgs == 0:
        print(f"No images in {img_dir}")
        return 1
    print(f"Comparison set: {img_dir} ({n_imgs} images)")
    print(f"Gemma URL: {args.gemma_url}")
    print("Models: gemma + qwen\n")

    # Run the verified comparison tool as a subprocess (both models)
    before = set(REPORTS.glob("vlm_compare_*.json"))
    cmd = [
        sys.executable, "-u", "-m", "api.module.clip_extraction.vlm.compare",
        "--models", "gemma,qwen",
        "--gemma-url", args.gemma_url,
        "--images", str(img_dir),
        "--max-tokens", str(args.max_tokens),
    ]
    print("running:", " ".join(cmd), "\n")
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        print(f"comparison exited rc={rc}")
        return rc

    # Find the new report and summarize
    after = set(REPORTS.glob("vlm_compare_*.json"))
    new = sorted(after - before)
    if new:
        from api.module.clip_extraction.vlm.summarize_vlm_compare import summarize
        print("\n" + "=" * 70)
        summarize(str(new[-1]), labels_csv=str(SET_DIR / "_labels.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
