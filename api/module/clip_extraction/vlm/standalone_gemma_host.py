#!/usr/bin/env python3
"""
Standalone VLM attribute-extraction comparison — RUN THIS ON THE GEMMA HOST.

Why standalone: TCP 8001 on the Gemma host (110.10.125.88) is blocked by an
upstream firewall between the Soongsil (203.253.x) and KT (110.10.x) networks,
so the comparison can't reach Gemma from another machine. Running here uses
localhost:8001 directly — no networking, no firewall, no tunnel.

This one file has NO project dependencies. Copy it (and a folder of test
images) to the Gemma host and run it there.

Requirements (likely already in your officevault_env; else pip install):
    pip install openai pillow

Usage
-----
    # Gemma only (default)
    python standalone_gemma_host.py --images ./sample_works

    # Gemma + Qwen3-VL side-by-side (set the key first; this box has internet)
    export DASHSCOPE_API_KEY=sk-xxxxxxxx
    python standalone_gemma_host.py --images ./sample_works --with-qwen

    # Point at a different Gemma URL / model if needed
    python standalone_gemma_host.py --images ./imgs --gemma-url http://127.0.0.1:8001/v1

Output
------
    vlm_compare_{TIMESTAMP}.json   — full per-image data
    vlm_compare_{TIMESTAMP}.md     — human-readable side-by-side report
written next to this script. Send me the .md (or .json) and I'll fold it into
the comparison.

Filename convention (optional): name images "{label}__name.jpg" (e.g.
"사진저작물__beach.jpg") and the report will note the expected work_type.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------- config ---
GEMMA_DEFAULT_URL = "http://127.0.0.1:8001/v1"
GEMMA_MODEL = "google/gemma-4-31B-it"
DASHSCOPE_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3-vl-235b-a22b-instruct"
MAX_LONG_EDGE = 1536  # px — downscale cap (keeps files under provider limits)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ----------------------------------------------------------------- prompt ---
# Keep identical to api/module/clip_extraction/vlm/prompts.py so results match.
SYSTEM_PROMPT = (
    "당신은 공유저작물(공공저작물·CCL·퍼블릭 도메인) 메타데이터 추출 전문가입니다. "
    "주어진 이미지를 분석하여 저작물의 속성 정보를 정확하게 추출합니다. "
    "반드시 유효한 JSON 객체 하나만 출력하고, 그 외의 설명·코드블록·마크다운은 절대 출력하지 마세요."
)

USER_PROMPT = """이 이미지를 분석하여 아래 JSON 스키마에 맞게 공유저작물 메타데이터를 추출하세요.

{
  "description": "이미지 내용을 한국어로 2~3문장으로 객관적으로 설명",
  "work_type": "아래 목록 중 '매체(medium)' 기준으로 하나만 선택",
  "work_type_reason": "그 매체로 판단한 근거 한 문장",
  "keywords": ["핵심 키워드 5~7개"],
  "main_subjects": ["이미지에 보이는 주요 객체/피사체 목록"],
  "dominant_colors": ["주요 색상 2~3개 (한국어)"],
  "text_in_image": "이미지 안에 보이는 글자(있으면 그대로, 없으면 null)",
  "scene_type": "실내/실외/스튜디오/그래픽 등",
  "estimated_quality": "고화질/중간/저화질 중 하나"
}

work_type 목록: 사진저작물, 영상저작물, 어문저작물, 음악저작물, 미술저작물, 건축저작물, 도형저작물, 컴퓨터프로그램저작물, 연극저작물, 기타

★ 매우 중요: work_type은 '무엇을 찍었는가(피사체)'가 아니라 '어떤 매체로 만들어졌는가'로 판단하세요.
  - 건물을 카메라로 촬영한 사진 → "사진저작물" (O), "건축저작물" (X)
  - 손으로 그린 그림/회화 → "미술저작물"
  - 지도·도표·설계도 → "도형저작물"
  - 동영상의 한 장면(프레임) → "영상저작물"

JSON 객체 하나만 출력하세요."""


# ----------------------------------------------------------------- helpers ---
def encode_image(path: Path) -> str:
    """Read, downscale if huge, return base64 JPEG data URL."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / max(im.size)
            im = im.resize((round(im.width * scale), round(im.height * scale)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    cand = fence.group(1) if fence else text
    if not fence:
        s, e = cand.find("{"), cand.rfind("}")
        if s != -1 and e > s:
            cand = cand[s:e + 1]
    try:
        return json.loads(cand)
    except (json.JSONDecodeError, ValueError):
        return None


def gt_from_filename(path: Path) -> str | None:
    return path.stem.split("__", 1)[0] if "__" in path.stem else None


def call_model(client, model: str, image_url: str, max_tokens: int) -> dict:
    """One image → one model. Returns timing + parsed result."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0,
        )
        dt = round(time.perf_counter() - t0, 2)
        text = resp.choices[0].message.content or ""
        parsed = extract_json(text)
        ctoks = resp.usage.completion_tokens if resp.usage else None
        return {"ok": True, "latency_s": dt, "raw_text": text,
                "parsed": parsed, "parse_ok": parsed is not None,
                "completion_tokens": ctoks}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "latency_s": round(time.perf_counter() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}


def build_report_md(report: dict) -> str:
    L = [f"# VLM Comparison (on Gemma host) — {report['timestamp']}", "",
         f"- Models: {', '.join(report['models'])}",
         f"- Images: {report['n_images']}", ""]
    L += ["## Summary", "",
          "| Model | OK | JSON parse OK | work_type acc | Avg latency | Avg tokens |",
          "|---|---|---|---|---|---|"]
    for label, a in report["summary"].items():
        acc = f"{a['acc']*100:.0f}%" if a["acc"] is not None else "n/a"
        L.append(f"| {label} | {a['ok']}/{report['n_images']} | "
                 f"{a['parsed']}/{report['n_images']} | {acc} | "
                 f"{a['avg_latency']}s | {a['avg_tokens']} |")
    L += ["", "## Per-image extraction"]
    for img in report["images"]:
        L += ["", f"### {img}"]
        gt = report["gt"].get(img)
        if gt:
            L.append(f"_expected work_type: {gt}_")
        for label in report["models"]:
            rec = report["by_image"][img].get(label)
            if not rec:
                continue
            L.append("")
            if not rec["ok"]:
                L.append(f"**{label}** — ERROR: {rec['error']}")
                continue
            p = rec.get("parsed") or {}
            L.append(f"**{label}** ({rec['latency_s']}s):")
            if p:
                L.append(f"- work_type: **{p.get('work_type','—')}** ({p.get('work_type_reason','')})")
                L.append(f"- description: {p.get('description','—')}")
                for k in ("keywords", "main_subjects", "dominant_colors"):
                    v = p.get(k) or []
                    L.append(f"- {k}: {', '.join(v) if isinstance(v, list) else v}")
                if p.get("text_in_image"):
                    L.append(f"- text_in_image: {p['text_in_image']}")
            else:
                L.append(f"- (JSON parse failed) raw: {rec.get('raw_text','')[:400]}")
    return "\n".join(L)


# ------------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone VLM comparison (run on Gemma host)")
    ap.add_argument("--images", required=True, help="Directory of test images")
    ap.add_argument("--gemma-url", default=GEMMA_DEFAULT_URL)
    ap.add_argument("--gemma-model", default=GEMMA_MODEL)
    ap.add_argument("--with-qwen", action="store_true",
                    help="Also run Qwen3-VL via DashScope (needs DASHSCOPE_API_KEY)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("Missing dep. Run: pip install openai pillow")
        return 1

    image_dir = Path(args.images)
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS) \
        if image_dir.exists() else []
    if not images:
        print(f"No images in {image_dir}")
        return 1

    # Build clients
    backends = []
    gemma = OpenAI(base_url=args.gemma_url, api_key="not-needed", timeout=300, max_retries=1)
    backends.append(("Gemma 4 31B (vLLM)", gemma, args.gemma_model))
    if args.with_qwen:
        key = os.getenv("DASHSCOPE_API_KEY", "")
        if not key:
            print("--with-qwen set but DASHSCOPE_API_KEY is empty. Skipping Qwen.")
        else:
            qwen = OpenAI(base_url=DASHSCOPE_BASE, api_key=key, timeout=300, max_retries=1)
            backends.append(("Qwen3-VL-235B (DashScope)", qwen, QWEN_MODEL))

    # Reachability check
    print("Backends:")
    live = []
    for label, client, model in backends:
        try:
            client.models.list()
            print(f"  ✓ {label}")
            live.append((label, client, model))
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {label}: {type(e).__name__}: {e}")
    if not live:
        print("No reachable backends — aborting.")
        return 2

    print(f"\nProcessing {len(images)} images...\n")
    by_image, gt_map = {}, {}
    agg = {label: {"ok": 0, "parsed": 0, "correct": 0, "n_gt": 0, "lat": [], "tok": []}
           for label, _, _ in live}

    for path in images:
        gt = gt_from_filename(path)
        gt_map[path.name] = gt
        url = encode_image(path)
        by_image[path.name] = {}
        print(f"=== {path.name} ===")
        for label, client, model in live:
            rec = call_model(client, model, url, args.max_tokens)
            by_image[path.name][label] = rec
            a = agg[label]
            if rec["ok"]:
                a["ok"] += 1
                a["lat"].append(rec["latency_s"])
                if rec.get("completion_tokens"):
                    a["tok"].append(rec["completion_tokens"])
                if rec["parse_ok"]:
                    a["parsed"] += 1
                    wt = (rec["parsed"] or {}).get("work_type", "?")
                    if gt is not None:
                        a["n_gt"] += 1
                        if wt == gt:
                            a["correct"] += 1
                    print(f"  ✓ {label}: work_type={wt} ({rec['latency_s']}s)")
                else:
                    print(f"  ⚠ {label}: ran but JSON parse failed ({rec['latency_s']}s)")
            else:
                print(f"  ✗ {label}: {rec['error']}")

    summary = {}
    for label, a in agg.items():
        summary[label] = {
            "ok": a["ok"], "parsed": a["parsed"],
            "acc": round(a["correct"] / a["n_gt"], 4) if a["n_gt"] else None,
            "avg_latency": round(sum(a["lat"]) / len(a["lat"]), 2) if a["lat"] else None,
            "avg_tokens": round(sum(a["tok"]) / len(a["tok"])) if a["tok"] else None,
        }

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": [label for label, _, _ in live],
        "n_images": len(images),
        "images": [p.name for p in images],
        "gt": gt_map, "summary": summary, "by_image": by_image,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent
    (out_dir / f"vlm_compare_{ts}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"vlm_compare_{ts}.md").write_text(
        build_report_md(report), encoding="utf-8")
    print(f"\nReports written next to this script:\n"
          f"  vlm_compare_{ts}.json\n  vlm_compare_{ts}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
