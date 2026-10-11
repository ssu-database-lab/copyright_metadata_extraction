#!/usr/bin/env python
"""설명(description) 정답 초안 생성기 — 사이트에 설명이 없거나 컬렉션 공통 문구뿐인 저작물에 대해,
평가 대상 시스템(Qwen·Gemma)과 다른 계열의 모델로 설명 초안을 만들고 사람 검수용 워크북을 만든다.

기준 문서: docs/test_prep_investigations_20261007/04_description_gold_and_guideline.md §3 (지침 v0.1),
           08_description_pilot/README.md (모델·프레임 수·임계값 근거).

사용 예
  # 1) 선정·입력 준비만 (API 호출 없음) — 프롬프트 예시, 비용 추정, dry_run.jsonl
  python api/module/dataset_builder/generate_descriptions.py --select missing_or_label --per-cell 25 --dry-run --out dataset/gold_descriptions/pilot_200
  # 2) 파일럿 200건 (이미지·영상 × 4 권리유형 × 25) — 같은 --out 에 실제 실행
  python api/module/dataset_builder/generate_descriptions.py --select missing_or_label --per-cell 25 --out dataset/gold_descriptions/pilot_200
  # 3) 확정된 시험 세트의 id 목록으로 (첫 열 = work_id, 헤더 유무 무관)
  python api/module/dataset_builder/generate_descriptions.py --ids test_set_ids.csv --out dataset/gold_descriptions/t002
  # 4) 중단 후 이어서 — 최초 실행 때 저장한 selected_ids.csv 를 그대로 쓴다 (인덱스가 바뀌어도 표본 불변)
  python ... --out dataset/gold_descriptions/pilot_200 --resume
  # 5) Kanana 로컬 교차 초안 추가 (GPU, 선택) / 검수 결과 반영
  python ... --out dataset/gold_descriptions/pilot_200 --resume --kanana
  python ... --out dataset/gold_descriptions/pilot_200 --import-review dataset/gold_descriptions/pilot_200/review.xlsx

출력 (--out 디렉터리)
  selected_ids.csv    최초 실행 때 확정한 표본 (resume 시 재사용)
  drafts.jsonl        저작물 1건 = 1행: 모델 원문·파싱 결과·자동검증·provenance (검수 전)
  kanana.jsonl        (--kanana) 교차 초안 체크포인트
  frames/<id>/        모델에 보낸 이미지/프레임(검수자가 그대로 봄) + 영상 contact sheet
  review.xlsx         검수 워크북 (verdict ACCEPT / EDIT / REJECT, 사유 코드, final_text)
  gold_descriptions.jsonl  (--import-review) 검수 반영 최종 정답 (gold_source=ai_draft_human_verified, tier=A-검수)
  prompt_system.txt, prompt_user_template.txt, run_manifest.json, run.log

API 키: 프로젝트 루트 .env 의 OPENAI_API_KEY, 환경변수, 또는 --env 로 지정한 파일. 키는 어디에도 기록하지 않는다.
Kanana(선택): `pip install timm` 필요. 모델의 remote code 가 vision encoder 에 flash_attention_2 를 하드코딩하고 fallback 도 실패하므로
  HF 캐시의 modeling.py 에서 attn_implementation="flash_attention_2" → "sdpa" 로 바꿔야 한다 (이 워크스테이션에는 적용됨).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "dataset" / "gt_merged_index.csv"
GUIDELINE_VERSION = "desc-guideline-v0.3"  # v0.2: 글자 수 하한 40→30, 해시태그 동시 생성 · v0.3: 제목의 주제어·수어 단어를 설명에 명시
PROMPT_ID = "desc-v0.3-json"
MIN_CHARS, MAX_CHARS = 30, 200
HASHTAG_MIN, HASHTAG_MAX = 5, 10
PRICE_USD_PER_M = {"gpt-5.4-mini-2026-03-17": (0.75, 4.5), "gpt-5.4-mini": (0.75, 4.5), "gpt-5.4": (2.5, 15.0)}  # (input, output), 2026-10 list
OUTPUT_TOKENS_BY_EFFORT = {"low": 700, "medium": 1800, "high": 3500}  # 답변 ~300 + 추론 토큰 (파일럿 실측 기반)
LABEL_MIN_ROWS = 3  # 같은 설명이 같은 media×bucket×source 안에서 3행 이상이면 컬렉션 라벨 (04 §1.5)
STOP_TAGS = re.compile(r"기증|릴레이|^비디오$|^영상$|^4K$|촬영$|저작물|^해양유물$|^수중문화재$")
ATTRIB_ONLY = re.compile(r"^.{1,40}의 (국외 )?(일러스트 작품|작품|사진 자료|자료)$")  # build_ground_truth._ATTRIB_ONLY + '국외' 변형
NO_TITLE = {"", "제목없음", "제목 없음", "untitled", "Untitled"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
MAX_CONSECUTIVE_ERRORS = 5

# ---------------------------------------------------------------- prompts (지침 v0.1 — 파일럿 pilot.py 와 문구 동일)
SYSTEM_PROMPT = """너는 공공·공유 저작물 아카이브의 설명문 작성자다. 주어진 이미지(또는 한 영상에서 시간순으로 뽑은 프레임들)에 실제로 보이는 것만을 한국어로 서술한다.
규칙:
1. 1~3문장, 공백 포함 30~200자, 현재형 평서문 '-다' 종결. '-입니다' 금지. "이 이미지는/영상은"으로 시작하지 않는다.
2. 첫 문장 = 주 피사체 + 장소/배경 + 동작·상태. 이후 문장 = 구체 정보(구도·시점, 색·조명, 화면 내 글자, 영상은 시간에 따른 변화).
3. 화면에서 확인할 수 없는 사실(제작연도, 작가, 역사·유래, 문화재 지정, 소장처, 행사 일정, 라이선스·출처·기증·수상)은 쓰지 않는다.
4. 고유명사는 아래 [제목]·[태그]에 있고 화면과 모순되지 않는 장소·시설·행사·유물명만 쓴다. 그 밖의 인물·작품·지명 이름은 지어내지 않고 일반 명사로 쓴다. 개인의 이름·나이는 절대 쓰지 않는다.
5. 평가·감상 어휘(아름다운, 멋진, 평화로운, 이색적인)와 추측 표현(~로 보인다)을 쓰지 않는다. 불확실하면 상위어를 쓴다.
6. 영상: 줄거리를 쓰지 않는다. 주 장면 1개와 시작→끝의 변화(카메라·피사체 이동, 장면 전환)를 적는다. 자막·타이틀카드의 글자만 인용할 수 있다. 음성은 알 수 없으므로 언급하지 않는다.
6-1. 제목이 저작물의 주제어(대상·단어·장소)를 알려주면 화면과 모순되지 않는 한 그 주제어를 설명과 해시태그에 넣는다. 수어 영상은 제목의 단어를 「'<단어>'를 수어로 표현한다」 형식으로 적고 손의 위치·움직임 1가지를 더한다. 컬렉션 안의 다른 저작물에도 그대로 들어맞는 일반 문구로 끝내지 않는다.
7. 해시태그: 화면에 보이는 대상·장소·행위·매체 특성을 나타내는 명사형 태그 5~10개를 "#태그" 형식으로 쓴다. 띄어쓰기 없이 붙여 쓰고, 제목·태그의 고유명사는 화면과 모순되지 않을 때만 쓴다. 개인의 이름·나이, 라이선스·출처·기증·공모전 관련 단어, 평가어는 쓰지 않는다. 설명과 같은 근거(화면)에서만 뽑는다.
8. 출력은 JSON 객체 하나만."""
USER_TEMPLATE = """매체: {media}
사이트/버킷: {source} / {bucket}
제목: {title}
태그(내용 태그만; 컬렉션 공통 태그 제외): {tags}
{video_line}
위 규칙에 따라 설명을 작성하고 다음 JSON으로만 답하라:
{{"description": "<1~3문장, 30~200자>", "hashtags": ["#<태그1>", "#<태그2>", "...(5~10개)"], "sentence_count": <정수>, "char_count": <정수>, "entities_from_title_or_tags": ["<설명에 사용한 고유명사>"], "visible_text": ["<화면에서 읽은 글자, 없으면 빈 배열>"], "uncertain_points": ["<확신이 없어 상위어로 쓴 항목>"], "unusable": {{"flag": false, "reason": null}}}}"""
FIX_TEMPLATE = "직전 답변은 다음 규칙을 어겼다: {problems}. 같은 JSON 형식으로 다시 작성하라."
PROMPT_HASH = hashlib.sha256((SYSTEM_PROMPT + USER_TEMPLATE + FIX_TEMPLATE).encode()).hexdigest()


class FatalAPIError(RuntimeError):
    """키 오류·잔액 소진처럼 재시도해도 소용없는 오류 — 실행 전체를 멈춘다."""


# ---------------------------------------------------------------- helpers
def load_env(path: Path, log=None) -> dict:
    d = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"(?:export\s+)?([A-Z0-9_]+)=(.*)$", ln.strip())
            if m:
                if m.group(1) in d and log:
                    log.warning("%s defines %s more than once — using the last definition", path.name, m.group(1))
                d[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return d


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_count(duration_s: float) -> int:
    """시스템(keyframes.py: 3/5/8/12/16)의 상위집합 — 08 §5c 권고."""
    if duration_s < 10:
        return 4
    if duration_s < 60:
        return 8
    if duration_s < 300:
        return 16
    if duration_s < 900:
        return 24
    return 32


def ffprobe_duration(p: Path, fallback: float | None) -> float:
    """영상 스트림 길이 → format.duration → 다른 스트림 최대 → 인덱스 값(ffprobe 실측, video_duration_sec). swf/flv 등은 format.duration 이 없다."""
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(p)], capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    streams = j.get("streams", [])
    cands = [float(s["duration"]) for s in streams if s.get("codec_type") == "video" and s.get("duration")]
    if not cands and j.get("format", {}).get("duration"):
        cands = [float(j["format"]["duration"])]
    if not cands:
        cands = [float(s["duration"]) for s in streams if s.get("duration")]
    if cands:
        return max(cands)
    if fallback:
        return float(fallback)
    raise ValueError("duration unknown (ffprobe and index)")


def to_rgb8(im: Image.Image) -> Image.Image:
    """16비트·부동 → 8비트, 투명 → 흰 배경 합성, 그 외 → RGB."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        import numpy as np
        a = np.asarray(im, dtype=np.float64)
        hi = max(a.max(), 1.0)
        scale = 257.0 if hi > 255 and hi <= 65535 and im.mode != "F" else (255.0 / hi if hi > 255 else 1.0)
        im = Image.fromarray((a / scale).clip(0, 255).astype("uint8"), "L")
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


PPTX_PS1 = r"""param([string]$In, [string]$Out)
$app = New-Object -ComObject PowerPoint.Application
try { $pres = $app.Presentations.Open($In, $true, $false, $false); $pres.Slides.Item(1).Export($Out, "PNG", 1920, 1080); $pres.Close(); "OK" } finally { $app.Quit() }
"""


def to_windows_path(p: Path) -> str:
    out = subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else str(p)


def render_pptx_first_slide(src: Path, dst_png: Path) -> None:
    """PPT 템플릿 저작물(사이트가 '이미지'로 분류)은 1쪽 슬라이드를 PowerPoint COM 으로 렌더링해 이미지로 쓴다 (1920×1080 PNG)."""
    tmp = Path("/mnt/c/Users/user/AppData/Local/Temp/pptx_render")
    tmp.mkdir(parents=True, exist_ok=True)
    ps1 = tmp / "render.ps1"
    ps1.write_bytes(b"\xef\xbb\xbf" + PPTX_PS1.encode("utf-8"))
    out_png = tmp / f"{src.stem[:40]}.png"
    r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", to_windows_path(ps1), "-In", to_windows_path(src), "-Out", to_windows_path(out_png)], capture_output=True, text=True, timeout=180)
    if not out_png.exists():
        raise RuntimeError(f"PowerPoint render failed: {r.stdout[-120:]} {r.stderr[-120:]}")
    dst_png.write_bytes(out_png.read_bytes())
    out_png.unlink(missing_ok=True)


def prep_image(src: Path, dst: Path, max_side: int) -> dict:
    if not dst.exists():
        im = to_rgb8(ImageOps.exif_transpose(Image.open(src)))
        w, h = im.size
        im.thumbnail((max_side, max_side))
        im.save(dst, "JPEG", quality=90)
    with Image.open(src) as o, Image.open(dst) as s:
        return {"orig_size": list(o.size), "sent_size": list(s.size)}


def prep_video(src: Path, out_dir: Path, max_side: int, fallback_dur: float | None) -> dict:
    dur = ffprobe_duration(src, fallback_dur)
    n = frame_count(dur)
    times = [round(0.3 + (dur - 0.6) * i / (n - 1), 2) for i in range(n)] if dur > 1.2 else [0.0]
    frames, kept = [], []
    scale = f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    for i, t in enumerate(times):
        p = out_dir / f"f{i:02d}_{t:.1f}s_{max_side}.jpg"
        for tt in (t, max(0.0, t - 0.5), max(0.0, t - 1.5)):  # 마지막 프레임 뒤(오디오 꼬리, VFR)로 seek 하면 파일이 안 생긴다 → 앞으로 당겨 재시도
            if p.exists():
                break
            subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-ss", str(tt), "-i", str(src), "-frames:v", "1", "-vf", scale, "-q:v", "3", str(p)])
        if p.exists():
            frames.append(p)
            kept.append(t)
    method = "seek"
    if len(frames) < n or src.suffix.lower() in (".swf", ".flv"):
        # SWF/FLV 는 -ss 시크가 신뢰할 수 없다(keyframes.py 와 같은 처리): 순차 디코드로 균등 간격 추출
        interval = max(dur / n, 0.05)
        seq = sorted(out_dir.glob(f"s*_{max_side}.jpg"))
        if len(seq) < min(n, 2):
            subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", str(src), "-vf", f"fps=1/{interval:.4f},{scale}", "-frames:v", str(n), "-q:v", "3", str(out_dir / f"s%02d_{max_side}.jpg")])
            seq = sorted(out_dir.glob(f"s*_{max_side}.jpg"))
        if len(seq) > len(frames):
            for p in frames:
                p.unlink(missing_ok=True)
            frames, kept, method = seq, [round(i * interval, 1) for i in range(len(seq))], "sequential"
            (out_dir / f"sheet_{max_side}.png").unlink(missing_ok=True)
    if not frames:
        raise ValueError("no frame could be extracted")
    sheet = out_dir / f"sheet_{max_side}.png"
    if not sheet.exists():
        cols, tw, th = 4, 320, 180
        rows = (len(frames) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * tw, rows * th), "black")
        for i, (p, t) in enumerate(zip(frames, kept)):
            im = Image.open(p).convert("RGB")
            im.thumbnail((tw, th))
            x, y = (i % cols) * tw, (i // cols) * th
            canvas.paste(im, (x, y))
            ImageDraw.Draw(canvas).text((x + 5, y + 5), f"{t:.1f}s", fill="yellow")
        canvas.save(sheet)
    return {"duration_s": round(dur, 1), "n_frames": len(frames), "frame_times": [round(t, 1) for t in kept], "frames_dropped": n - len(frames), "extraction": method, "frames": [str(p) for p in frames], "sheet": str(sheet)}


def data_url(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def parse_json(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


_QUOTED = re.compile(r"「[^」]*」|“[^”]*”|\"[^\"]*\"|'[^']*'")
_PARTICLE = re.compile(r"(의|에서|으로|로|을|를|이|가|은|는|과|와|도)$")


def _norm_entity(s: str) -> str:
    s = re.sub(r"[\s·_\-,.()\[\]「」'\"“”/]", "", str(s).casefold())
    s2 = _PARTICLE.sub("", s)
    return s2 if len(s2) >= 2 else s


def check_guideline(desc: str, title: str, tags: list[str], js: dict | None) -> dict:
    d = (desc or "").strip()
    masked = _QUOTED.sub("□", d)  # 규칙 6: 화면 글자 인용은 허용 → 문체 검사에서 제외
    tail = re.sub(r"[」”\"')\]\s]+$", "", d)
    sents = [s for s in re.split(r"(?<=[.!?。])[」”\"')\]]*\s+", masked) if s.strip()]
    problems = []
    if not d:
        problems.append("설명 없음")
    if d and not (MIN_CHARS <= len(d) <= MAX_CHARS):
        problems.append(f"글자 수 {len(d)} ({MIN_CHARS}~{MAX_CHARS} 필요)")
    if d and not (1 <= len(sents) <= 3):
        problems.append(f"문장 수 {len(sents)} (1~3 필요)")
    if d and not tail.endswith("다."):
        problems.append("'-다.' 종결 아님")
    if re.search(r"입니다|습니다", masked):
        problems.append("'-입니다' 사용")
    if re.match(r"^(이|본)\s*(이미지|영상|사진)[은는]", d):
        problems.append("'이 이미지는/영상은' 시작")
    if re.search(r"로 보인다|처럼 보인다|로 추정|인 듯|듯하다|것 같다", masked):
        problems.append("추측 표현")
    if re.search(r"아름다운|멋진|평화로운|이색적인|인상적인", masked):
        problems.append("평가 어휘")
    pool = _norm_entity(title if title not in NO_TITLE else "") + "|" + "|".join(_norm_entity(t) for t in tags)
    ents = (js or {}).get("entities_from_title_or_tags") or []
    if isinstance(ents, str):
        ents = [ents]
    bad = []
    for e in ents:
        if not isinstance(e, str):
            continue
        for part in re.split(r"[·,/]", e):
            n = _norm_entity(part)
            if n and n not in pool:
                bad.append(part.strip())
    if bad:
        problems.append(f"제목·태그에 없는 고유명사 {bad}")
    tags_out = normalize_hashtags((js or {}).get("hashtags"))
    if js is not None and d:
        if not (HASHTAG_MIN <= len(tags_out) <= HASHTAG_MAX):
            problems.append(f"해시태그 수 {len(tags_out)} ({HASHTAG_MIN}~{HASHTAG_MAX} 필요)")
        badtags = [t for t in tags_out if not re.fullmatch(r"#[0-9A-Za-z가-힣ㄱ-ㅎ·_]{1,15}", t)]
        if badtags:
            problems.append(f"해시태그 형식 오류 {badtags[:5]}")
        forbidden = [t for t in tags_out if re.search(r"기증|라이선스|저작권|공유마당|공공누리|KOGL|공모전|입선|아름다운|멋진", t)]
        if forbidden:
            problems.append(f"해시태그 금지어 {forbidden}")
    return {"chars": len(d), "sentences": len(sents), "problems": problems, "ok": not problems, "hashtags": tags_out}


def normalize_hashtags(raw) -> list[str]:
    """문자열/리스트 → '#태그' 목록 (공백 제거, '#' 보강, 중복 제거, 순서 유지)."""
    if raw is None:
        return []
    items = re.split(r"[\s,]+", raw) if isinstance(raw, str) else [str(x) for x in raw if x is not None]
    out, seen = [], set()
    for t in items:
        t = re.sub(r"\s+", "", t.strip())
        if not t:
            continue
        t = "#" + t.lstrip("#")
        key = t.casefold()
        if key not in seen and len(t) > 1:
            seen.add(key)
            out.append(t)
    return out


def clean_cell(v):
    """openpyxl 이 거부하는 제어문자 제거 (수식 오인은 data_type 으로 막는다)."""
    if isinstance(v, str):
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", v)
    return v


# ---------------------------------------------------------------- selection
def classify_site_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """site_kind: none | label | caption | plot | background (04 §1.5, §3d site_gold_kind)."""
    tierA = df["work_설명_tier"] == "A"
    counts = df[tierA].groupby(["media", "license_bucket", "source", "work_설명"]).size()
    kinds = []
    for _, r in df.iterrows():
        s = r["work_설명"].strip()
        if r["work_설명_tier"] != "A" or not s or re.match(r"^https?://", s) or ATTRIB_ONLY.match(s):
            kinds.append("none")
        elif counts.get((r.media, r.license_bucket, r.source, r["work_설명"]), 0) >= LABEL_MIN_ROWS:
            kinds.append("label")
        elif r.media == "video" and r.license_bucket == "expired":
            kinds.append("plot")
        elif r.media == "video" and r.source.startswith("공공누리"):
            kinds.append("background")
        else:
            kinds.append("caption")
    df = df.copy()
    df["site_kind"] = kinds
    return df


SELECT_KINDS = {"missing": {"none"}, "missing_or_label": {"none", "label"}, "missing_label_plot": {"none", "label", "plot"}, "all": {"none", "label", "plot", "background", "caption"}}


def select_works(args, log=None) -> pd.DataFrame:
    df = pd.read_csv(INDEX, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    tf = lambda s: s.str.upper() == "TRUE"
    df = df[df.media.isin(args.media) & tf(df.work_file_exists) & tf(df.has_work_gt)]
    if args.source != "all":
        df = df[df.source == args.source]
    if args.bucket != "all":
        df = df[df.license_bucket.isin(args.bucket.split(","))]
    df = classify_site_descriptions(df)
    if args.ids:
        idf = pd.read_csv(args.ids, dtype=str, keep_default_na=False, header=None)
        if "work_id" in idf.iloc[0].tolist():  # 헤더가 있는 CSV → work_id 열
            col = idf.iloc[0].tolist().index("work_id")
            raw = [str(x).strip() for x in idf.iloc[1:, col] if str(x).strip()]
        else:  # 헤더 없는 id 목록 → 첫 열
            raw = [str(x).strip() for x in idf.iloc[:, 0] if str(x).strip()]
        known = set(df.work_id)
        ids = [x for x in raw if x in known]
        unmatched = [x for x in raw if x not in known]
        if unmatched and log:
            log.warning("%d ids not in the image/video pool with files (first: %s) — header row or text/missing works?", len(unmatched), unmatched[:5])
        df = df[df.work_id.isin(ids)]
    else:
        df = df[df.site_kind.isin(SELECT_KINDS[args.select])]
    rng = random.Random(args.seed)
    if args.per_cell:
        parts = []
        for _, g in df.groupby(["media", "license_bucket"]):
            idx = list(g.index)
            rng.shuffle(idx)
            parts.append(g.loc[idx[: args.per_cell]])
        df = pd.concat(parts) if parts else df.iloc[0:0]
    if args.limit:
        idx = list(df.index)
        rng.shuffle(idx)
        df = df.loc[idx[: args.limit]]
    return df.sort_values(["media", "license_bucket", "work_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- model calls
class Drafter:
    def __init__(self, args, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, timeout=300, max_retries=0)
        self.args = args

    def call(self, messages: list, max_tokens: int) -> dict:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
        delay = 5
        for attempt in range(1, 5):
            try:
                t0 = time.time()
                r = self.client.chat.completions.create(model=self.args.model, messages=messages, max_completion_tokens=max_tokens,
                                                        reasoning_effort=self.args.effort, response_format={"type": "json_object"})
                u = r.usage
                return {"text": r.choices[0].message.content or "", "finish_reason": r.choices[0].finish_reason, "request_id": getattr(r, "id", None),
                        "model_resolved": getattr(r, "model", None), "system_fingerprint": getattr(r, "system_fingerprint", None),
                        "elapsed_s": round(time.time() - t0, 1), "attempt": attempt,
                        "usage": {"prompt": u.prompt_tokens, "completion": u.completion_tokens,
                                  "reasoning": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)}}
            except RateLimitError as e:
                if "insufficient_quota" in str(e):
                    raise FatalAPIError("insufficient_quota — add credits") from e
                logging.warning("rate limit (attempt %d) — retry in %ds", attempt, delay)
            except (APITimeoutError, APIConnectionError) as e:
                logging.warning("transient %s (attempt %d) — retry in %ds", type(e).__name__, attempt, delay)
            except APIStatusError as e:
                if e.status_code in (401, 403) or "invalid_api_key" in str(e):
                    raise FatalAPIError(f"auth error {e.status_code}") from e
                if e.status_code >= 500:
                    logging.warning("server %s (attempt %d) — retry in %ds", e.status_code, attempt, delay)
                else:
                    raise
            if attempt < 4:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("API call failed after 4 attempts")


def user_text(rec: dict) -> str:
    tags = json.dumps(rec["tags_content"], ensure_ascii=False) if rec["tags_content"] else "없음"
    spec = rec["input_spec"]
    video_line = (f"프레임 {spec['n_frames']}장, 시점(초): {', '.join(f'{t:.1f}' for t in spec['frame_times'])}, 총 길이 {spec['duration_s']}초\n") if rec["media"] == "video" else ""
    title = rec["title"] if rec["title"] not in NO_TITLE else "(제목 없음)"
    return USER_TEMPLATE.format(media=rec["media"], source=rec["source"], bucket=rec["bucket"], title=title, tags=tags, video_line=video_line)


def build_messages(rec: dict, frames: list[Path]) -> list:
    content = [{"type": "text", "text": user_text(rec)}] + [{"type": "image_url", "image_url": {"url": data_url(p)}} for p in frames]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def cost_usd(model: str, usage: dict) -> float:
    pi, po = PRICE_USD_PER_M.get(model, PRICE_USD_PER_M["gpt-5.4-mini"])
    return round(((usage.get("prompt") or 0) * pi + (usage.get("completion") or 0) * po) / 1e6, 5)


def choose_best(attempts: list[dict]) -> dict | None:
    """설명이 있는 시도 중 규칙 위반이 가장 적은 것 (동률이면 나중 것)."""
    cands = [a for a in attempts if a["draft_text"]]
    if not cands:
        return None
    return sorted(cands, key=lambda a: (len(a["checks"]["problems"]), -attempts.index(a)))[0]


# ---------------------------------------------------------------- per-work processing
def process(row: pd.Series, args, drafter: Drafter | None, out: Path) -> dict:
    wid = row.work_id
    rec = {"work_id": wid, "media": row.media, "source": row.source, "bucket": row.license_bucket, "title": row.title,
           "tags_all": [t.strip() for t in row["work_키워드"].split("|") if t.strip()], "tags_batch_level": row["work_키워드_batch_level"] == "True",
           "site_raw": row["work_설명"] if row["work_설명"] else (row.get("work_설명_raw") or ""), "site_tier": row["work_설명_tier"], "site_note": row.get("work_설명_note", ""),
           "site_gold_kind": row.site_kind, "file": row.work_file, "gold_source": "ai_draft", "tier": "A-검수(대기)", "draft_provider": "openai",
           "guideline_version": GUIDELINE_VERSION, "prompt_id": PROMPT_ID, "prompt_hash": PROMPT_HASH, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    author_parts = {a.strip() for a in re.split(r"[,/|·;]|\s+", str(row.get("work_저자", ""))) if len(a.strip()) >= 2}  # 저자·기증자 이름은 앵커로 주지 않는다 (규칙 4)
    rec["tags_content"] = [] if rec["tags_batch_level"] else [t for t in rec["tags_all"] if not STOP_TAGS.search(t) and t not in author_parts]
    rec["prompt_inputs_hash"] = hashlib.sha256(f"{rec['media']}|{rec['source']}|{rec['bucket']}|{rec['title']}|{'|'.join(rec['tags_content'])}".encode()).hexdigest()[:16]
    src = Path(row.work_file)
    fdir = out / "frames" / wid
    fdir.mkdir(parents=True, exist_ok=True)
    rendered = None
    if row.media == "image" and src.suffix.lower() == ".pptx":
        rendered = fdir / "slide1.png"
        if not rendered.exists():
            render_pptx_first_slide(src, rendered)
    elif row.media == "image" and src.suffix.lower() not in IMAGE_EXT:
        rec.update({"status": "unsupported_format", "error": f"not an image file: {src.suffix}"})
        return rec
    if row.media == "image":
        p = fdir / f"img_{args.max_side}.jpg"
        meta = prep_image(rendered or src, p, args.max_side)
        frames = [p]
        rec["input_spec"] = {"max_side": args.max_side, "n_frames": 1, **meta, **({"rendered_from": "pptx slide 1 via PowerPoint COM, 1920x1080"} if rendered else {})}
    else:
        v = prep_video(src, fdir, args.max_side, float(row.video_duration_sec) if row.video_duration_sec else None)
        frames = [Path(f) for f in v.pop("frames")]
        rec["input_spec"] = {"max_side": args.max_side, **v}
    if args.hash:
        rec["file_sha256"] = sha256_file(src)
    rec["model"] = args.model
    rec["params"] = {"reasoning_effort": args.effort, "max_completion_tokens": args.max_tokens, "response_format": "json_object", "temperature": "n/a (reasoning model)"}
    messages = build_messages(rec, frames)
    if drafter is None:  # dry run
        rec["status"] = "dry_run"
        rec["prompt_preview"] = messages[1]["content"][0]["text"]
        return rec
    attempts, max_tokens, budget_retries, fix_rounds = [], args.max_tokens, 0, 0
    while True:
        res = drafter.call(messages, max_tokens)
        js = parse_json(res["text"])
        desc = ((js or {}).get("description") or "").strip() if js else ""
        flagged = bool(((js or {}).get("unusable") or {}).get("flag")) if js else False
        attempts.append({**res, "json": js, "draft_text": desc, "checks": check_guideline(desc, rec["title"], rec["tags_content"], js), "unusable_flag": flagged})
        if flagged:  # 모델이 식별 불가로 판정 — 재시도해도 같다
            break
        if (res["finish_reason"] == "length" or not desc) and budget_retries < 2:  # 추론 토큰이 예산을 다 쓴 경우
            budget_retries += 1
            max_tokens = int(max_tokens * 1.5)
            continue
        if desc and not attempts[-1]["checks"]["ok"] and fix_rounds < 1:
            fix_rounds += 1
            messages = messages[:2] + [{"role": "assistant", "content": res["text"]}, {"role": "user", "content": FIX_TEMPLATE.format(problems="; ".join(attempts[-1]["checks"]["problems"]))}]
            continue
        break
    best = choose_best(attempts)
    rec["attempts"] = attempts
    rec["usage"] = {k: sum((a["usage"].get(k) or 0) for a in attempts) for k in ("prompt", "completion", "reasoning")}
    rec["elapsed_s"] = round(sum(a["elapsed_s"] for a in attempts), 1)
    rec["cost_usd"] = cost_usd(args.model, rec["usage"])
    last = attempts[-1]
    rec.update({"request_id": (best or last)["request_id"], "system_fingerprint": (best or last)["system_fingerprint"], "model_resolved": (best or last)["model_resolved"]})
    flagged = next((a for a in attempts if a.get("unusable_flag")), None)
    if best is None:
        if flagged:
            rec.update({"status": "unusable", "unusable_reason": ((flagged["json"] or {}).get("unusable") or {}).get("reason"), "draft_text": "", "json": flagged["json"], "checks": flagged["checks"]})
        else:
            rec.update({"status": "no_output", "draft_text": "", "json": last["json"], "checks": last["checks"]})
    else:
        rec.update({"draft_text": best["draft_text"], "draft_hashtags": best["checks"].get("hashtags", []), "json": best["json"], "checks": best["checks"],
                    "status": "ok" if best["checks"]["ok"] else "needs_review"})
        if flagged:
            rec["unusable_reason"] = ((flagged["json"] or {}).get("unusable") or {}).get("reason")
    return rec


# ---------------------------------------------------------------- Kanana cross-check (optional, local GPU)
def kanana_pass(records: list[dict], out: Path, log):
    import torch
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
    MODEL = "kakaocorp/kanana-1.5-v-3b-instruct"
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    for c in [cfg] + [getattr(cfg, a) for a in dir(cfg) if a.endswith("_config") and hasattr(getattr(cfg, a, None), "to_dict")]:
        try:
            c._attn_implementation = "sdpa"
        except Exception:
            pass
    try:
        model = AutoModelForVision2Seq.from_pretrained(MODEL, config=cfg, trust_remote_code=True, attn_implementation="sdpa", dtype=torch.bfloat16, device_map="cuda").eval()
        mode = "bf16"
    except Exception as e:  # noqa: BLE001 — OOM 등이면 8bit 로
        log.warning("Kanana bf16 load failed (%s) — trying 8-bit", type(e).__name__)
        from transformers import BitsAndBytesConfig
        model = AutoModelForVision2Seq.from_pretrained(MODEL, config=cfg, trust_remote_code=True, attn_implementation="sdpa", quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda").eval()
        mode = "int8"
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    log.info("Kanana loaded (%s, %.1f GB)", mode, torch.cuda.memory_allocated() / 1e9)
    ck = out / "kanana.jsonl"
    done = set()
    if ck.exists():
        for ln in ck.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(ln)["work_id"])
            except Exception:
                pass
    for rec in records:
        if rec["work_id"] in done or rec.get("status") in ("error", "dry_run", "unsupported_format"):
            continue
        fdir = out / "frames" / rec["work_id"]
        frames = sorted(fdir.glob("f*.jpg")) if rec["media"] == "video" else [fdir / f"img_{rec['input_spec']['max_side']}.jpg"]
        imgs = [Image.open(f).convert("RGB") for f in frames]
        sample = {"image": imgs, "conv": [{"role": "user", "content": " ".join(["<image>"] * len(imgs))}, {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_text(rec)}]}
        row = {"work_id": rec["work_id"], "kanana_model": f"{MODEL} ({mode})"}
        try:
            inputs = proc.batch_encode_collate([sample], padding_side="left", add_generation_prompt=True, max_length=8192)
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                gens = model.generate(**inputs, max_new_tokens=400, do_sample=False, num_beams=1)
            n_in = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            text = proc.tokenizer.batch_decode(gens[:, n_in:] if n_in and gens.shape[1] > n_in else gens, skip_special_tokens=True)[0]
            js = parse_json(text)
            row["kanana_text"] = ((js or {}).get("description") or "").strip() if js else text.strip()
        except Exception as e:  # noqa: BLE001
            row["kanana_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            torch.cuda.empty_cache()
        with open(ck, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rec.update({k: v for k, v in row.items() if k != "work_id"})
    del model
    torch.cuda.empty_cache()


def merge_kanana(records: list[dict], out: Path):
    ck = out / "kanana.jsonl"
    if not ck.exists():
        return
    by = {}
    for ln in ck.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
            by[r["work_id"]] = r
        except Exception:
            pass
    for rec in records:
        if rec["work_id"] in by:
            rec.update({k: v for k, v in by[rec["work_id"]].items() if k != "work_id"})


# ---------------------------------------------------------------- review workbook + import
WB_COLS = ["work_id", "media", "source", "bucket", "title", "tags_content", "site_gold_kind", "site_설명(원문)", "site_키워드", "draft(GPT)", "draft_hashtags", "draft(Kanana)", "entities", "visible_text",
           "uncertain_points", "unusable_reason", "자동검증", "status", "duration_s", "n_frames", "frames_dir", "verdict", "final_text", "final_hashtags", "reason_codes", "verifier", "verified_at", "notes"]


def write_workbook(records: list[dict], out: Path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(WB_COLS)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for r in records:
        js = r.get("json") or {}
        spec = r.get("input_spec", {})
        draft = r.get("draft_text", "")
        prefill = draft if r.get("status") in ("ok", "needs_review") and draft else ""
        tags_draft = " ".join(r.get("draft_hashtags", []))
        vals = [r["work_id"], r["media"], r["source"], r["bucket"], r["title"], " | ".join(r.get("tags_content", [])), r.get("site_gold_kind", ""), r.get("site_raw", ""), " | ".join(r.get("tags_all", [])), draft, tags_draft, r.get("kanana_text", ""),
                json.dumps(js.get("entities_from_title_or_tags", []), ensure_ascii=False), json.dumps(js.get("visible_text", []), ensure_ascii=False), json.dumps(js.get("uncertain_points", []), ensure_ascii=False),
                r.get("unusable_reason", "") or "", "; ".join(r.get("checks", {}).get("problems", [])) or "OK", r.get("status", ""), spec.get("duration_s", ""), spec.get("n_frames", ""),
                str(out / "frames" / r["work_id"]), "", prefill, tags_draft if prefill else "", "", "", "", ""]
        ws.append([clean_cell(v) for v in vals])
        for c in ws[ws.max_row]:
            if isinstance(c.value, str) and c.value.startswith("="):
                c.data_type = "s"
    widths = {"work_id": 11, "title": 32, "tags_content": 28, "site_설명(원문)": 40, "site_키워드": 28, "draft(GPT)": 56, "draft_hashtags": 34, "final_hashtags": 34, "draft(Kanana)": 44, "entities": 20, "visible_text": 20, "uncertain_points": 20,
              "unusable_reason": 20, "자동검증": 26, "status": 13, "frames_dir": 30, "verdict": 10, "final_text": 56, "reason_codes": 14, "verifier": 12, "verified_at": 14, "notes": 28}
    for i, name in enumerate(WB_COLS, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(name, 10)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    vcol = get_column_letter(WB_COLS.index("verdict") + 1)
    dv = DataValidation(type="list", formula1='"ACCEPT,EDIT,REJECT"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{vcol}2:{vcol}{max(2, len(records) + 1)}")
    ws.freeze_panes = "F2"
    g = wb.create_sheet("guide")
    for line in ["검수 절차 (04_description_gold_and_guideline.md §3c)",
                 "1 원본 파일을 연다 (이미지: 원본 해상도; 영상 ≤60초 전체, >60초는 5지점 이상 스크럽). frames_dir 의 sheet_*.png 는 모델이 본 프레임이다.",
                 "2 verdict: ACCEPT = 수정 없음 / EDIT = 한 문장 이내 직접 수정 → final_text 에 쓴다 / REJECT = 핵심 오류 (재초안 큐). draft 열은 고치지 않는다 (provenance).",
                 "2-1 해시태그: draft_hashtags 를 보고 final_hashtags 에 확정 목록을 쓴다 (공백 구분, '#' 포함, 5~10개). 화면에 없는 대상·개인명·라이선스/기증 단어는 지운다. EDIT 판정은 설명·해시태그 중 하나만 고쳐도 EDIT.",
                 "사유 코드 (reason_codes, 쉼표 구분): 1 주피사체·장소·동작 불일치 | 2 화면에 없는 사물·인물·글자(환각) | 3 제목·태그 밖 고유명사 또는 개인명 | 4 배경사실·라이선스·출처·기술정보·추측·평가어 | 5 형식(문장수·글자수·종결) | 6 영상 전개 오류·줄거리·음성 언급 | 7 컬렉션 공통 문구 수준 | 8 unusable 판정 타당성",
                 "status: ok = 자동검증 통과 / needs_review = 규칙 위반 있음(자동검증 열 참고) / unusable = 모델이 식별 불가 판정 / no_output = 답 없음 (재실행 대상)",
                 "검수 결과 반영: python api/module/dataset_builder/generate_descriptions.py --out <이 폴더> --import-review review.xlsx → gold_descriptions.jsonl"]:
        g.append([line])
    g.column_dimensions["A"].width = 170
    wb.save(out / "review.xlsx")


def import_review(out: Path, xlsx: Path, log):
    from openpyxl import load_workbook
    drafts = {}
    for ln in (out / "drafts.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
            drafts[r["work_id"]] = r
        except Exception:
            pass
    ws = load_workbook(xlsx, data_only=True)["review"]
    hdr = [c.value for c in ws[1]]
    col = {name: hdr.index(name) for name in WB_COLS if name in hdr}
    n = {"ACCEPT": 0, "EDIT": 0, "REJECT": 0, "pending": 0}
    with open(out / "gold_descriptions.jsonl", "w", encoding="utf-8") as f:
        for row in ws.iter_rows(min_row=2, values_only=True):
            wid = str(row[col["work_id"]]).strip()
            verdict = (row[col["verdict"]] or "").strip().upper() if row[col["verdict"]] else ""
            if verdict not in ("ACCEPT", "EDIT", "REJECT"):
                n["pending"] += 1
                continue
            n[verdict] += 1
            d = drafts.get(wid, {})
            final = (row[col["final_text"]] or "").strip() if verdict != "REJECT" else ""
            final_tags = normalize_hashtags(row[col["final_hashtags"]]) if ("final_hashtags" in col and verdict != "REJECT") else []
            if verdict == "ACCEPT":
                final = d.get("draft_text", "") or final
                final_tags = d.get("draft_hashtags", []) or final_tags
            g = {"work_id": wid, "media": d.get("media"), "source": d.get("source"), "bucket": d.get("bucket"), "title": d.get("title"),
                 "description": final, "hashtags": final_tags, "gold_source": "ai_draft_human_verified" if verdict != "REJECT" else None, "tier": "A-검수" if verdict != "REJECT" else None,
                 "verification_outcome": verdict.lower(), "edited": verdict == "EDIT" and (final != d.get("draft_text", "") or final_tags != d.get("draft_hashtags", [])), "draft_text": d.get("draft_text", ""), "draft_hashtags": d.get("draft_hashtags", []),
                 "edit_reason_codes": [x.strip() for x in str(row[col["reason_codes"]] or "").split(",") if x.strip()], "verifier_id": row[col["verifier"]], "verified_at": str(row[col["verified_at"]] or ""),
                 "notes": row[col["notes"]], "site_raw": d.get("site_raw"), "site_gold_kind": d.get("site_gold_kind"), "draft_model": d.get("model_resolved") or d.get("model"), "request_id": d.get("request_id"),
                 "prompt_id": d.get("prompt_id"), "prompt_hash": d.get("prompt_hash"), "guideline_version": d.get("guideline_version"), "input_spec": d.get("input_spec")}
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    log.info("review imported: %s → %s", n, out / "gold_descriptions.jsonl")


# ---------------------------------------------------------------- main
def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", choices=list(SELECT_KINDS), default="missing_or_label")
    ap.add_argument("--ids", help="CSV/text whose first column is work_id (header optional); overrides --select")
    ap.add_argument("--media", nargs="+", default=["image", "video"], choices=["image", "video"])
    ap.add_argument("--source", default="all", help="all | 공유마당 | 공공누리(KOGL)")
    ap.add_argument("--bucket", default="all", help="all or comma list of ccl,donated,expired,kogl")
    ap.add_argument("--per-cell", type=int, default=0, help="sample N per media×bucket cell")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="output dir (default dataset/gold_descriptions/run_<timestamp>)")
    ap.add_argument("--env", default=str(ROOT / ".env"), help=".env file holding OPENAI_API_KEY")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="prepare inputs, print one prompt and an estimate; no API calls")
    ap.add_argument("--hash", action="store_true", help="also record SHA-256 of each work file (slow on E:)")
    ap.add_argument("--kanana", action="store_true", help="add a Kanana-1.5-V-3B cross draft (local GPU)")
    ap.add_argument("--import-review", help="merge verdicts from review.xlsx into gold_descriptions.jsonl and exit")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "dataset" / "gold_descriptions" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(out / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("gen")
    if args.import_review:
        import_review(out, Path(args.import_review), log)
        return
    (out / "prompt_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (out / "prompt_user_template.txt").write_text(USER_TEMPLATE + "\n\n[FIX]\n" + FIX_TEMPLATE, encoding="utf-8")
    if args.model not in PRICE_USD_PER_M:
        log.warning("no price table for %s — cost figures assume gpt-5.4-mini rates", args.model)

    # ---- sample: fixed at the first real run, reused on --resume (인덱스가 재생성돼도 표본 불변)
    sel_file = out / "selected_ids.csv"
    index_hash = sha256_file(INDEX)
    if args.resume and sel_file.exists():
        saved = json.loads((out / "selection.json").read_text(encoding="utf-8")) if (out / "selection.json").exists() else {}
        if saved.get("index_sha256") and saved["index_sha256"] != index_hash:
            log.warning("gt_merged_index.csv changed since the first run — reusing selected_ids.csv anyway")
        args.ids = str(sel_file)
        args.per_cell, args.limit = 0, 0
    df = select_works(args, log)
    if not args.dry_run and not (args.resume and sel_file.exists()):
        df[["work_id"]].to_csv(sel_file, index=False, header=False, encoding="utf-8")
        (out / "selection.json").write_text(json.dumps({"index_sha256": index_hash, "args": {k: v for k, v in vars(args).items() if k != "env"}, "n": len(df)}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("selected %d works: %s", len(df), df.groupby(["media", "license_bucket"]).size().to_dict())

    drafts = out / "drafts.jsonl"
    records_by_id = {}
    if drafts.exists():
        for ln in drafts.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                if r.get("status") not in ("error", "dry_run", "no_output", "unsupported_format"):
                    records_by_id[r["work_id"]] = r  # 같은 id 가 여러 번이면 마지막 것
            except Exception:
                pass
        if records_by_id and not args.resume and not args.dry_run:
            log.error("%s already has %d records — pass --resume to continue or use a new --out", drafts, len(records_by_id))
            sys.exit(2)
    todo = df[~df.work_id.isin(records_by_id)]
    est_in = est_out = 0
    for _, r in todo.iterrows():
        n = 1 if r.media == "image" else frame_count(float(r.video_duration_sec) if r.video_duration_sec else 30.0)
        est_in += 600 + 800 * n
        est_out += OUTPUT_TOKENS_BY_EFFORT[args.effort]
    pi, po = PRICE_USD_PER_M.get(args.model, PRICE_USD_PER_M["gpt-5.4-mini"])
    log.info("to process: %d (done before: %d) | estimate ≈ %.0fk input + %.0fk output tokens ≈ $%.2f (×1.3 with retries ≈ $%.2f)", len(todo), len(records_by_id), est_in / 1e3, est_out / 1e3,
             (est_in * pi + est_out * po) / 1e6, 1.3 * (est_in * pi + est_out * po) / 1e6)

    drafter = None
    if not args.dry_run:
        key = load_env(Path(args.env), log).get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            log.error("OPENAI_API_KEY not found in %s or the environment — add it there, export it, or pass --env <file>", args.env)
            sys.exit(2)
        drafter = Drafter(args, key)
    lock = threading.Lock()
    stop = threading.Event()
    new_records, consecutive_errors = [], [0]

    def work(row):
        if stop.is_set():
            return None
        try:
            rec = process(row, args, drafter, out)
            consecutive_errors[0] = 0
        except FatalAPIError as e:
            rec = {"work_id": row.work_id, "media": row.media, "title": row.title, "status": "error", "error": f"FatalAPIError: {e}"}
            log.error("%s — stopping the run", e)
            stop.set()
        except Exception as e:  # noqa: BLE001
            rec = {"work_id": row.work_id, "media": row.media, "title": row.title, "status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            log.error("%s: %s", row.work_id, rec["error"])
            consecutive_errors[0] += 1
            if consecutive_errors[0] >= MAX_CONSECUTIVE_ERRORS:
                log.error("%d consecutive errors — stopping the run", MAX_CONSECUTIVE_ERRORS)
                stop.set()
        with lock:
            with open(drafts, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            new_records.append(rec)
            n = len(new_records)
            if n % 10 == 0 or n == len(todo):
                log.info("%d/%d done | ok %d, needs_review %d, unusable %d, no_output %d, error %d | $%.3f", n, len(todo),
                         *(sum(1 for r in new_records if r.get("status") == s) for s in ("ok", "needs_review", "unusable", "no_output", "error")), sum(r.get("cost_usd", 0) for r in new_records))
        return rec

    if args.dry_run:
        with open(out / "dry_run.jsonl", "w", encoding="utf-8") as f:
            for _, r in todo.head(max(1, min(len(todo), 3))).iterrows():
                try:
                    rec = process(r, args, None, out)
                except Exception as e:  # noqa: BLE001
                    rec = {"work_id": r.work_id, "media": r.media, "status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new_records.append(rec)
        ex = next((r for r in new_records if r.get("prompt_preview")), None)
        if ex:
            log.info("example prompt for %s (%s):\n%s", ex["work_id"], ex["media"], ex["prompt_preview"])
        log.info("dry run: %d items prepared under %s, records in dry_run.jsonl (no API calls)", len(new_records), out / "frames")
    else:
        ex = ThreadPoolExecutor(max_workers=args.workers)
        futs = [ex.submit(work, r) for _, r in todo.iterrows()]
        try:
            for fu in futs:
                fu.result()
        except KeyboardInterrupt:
            log.warning("interrupted — finishing in-flight items, cancelling the rest")
            stop.set()
            ex.shutdown(wait=True, cancel_futures=True)
        else:
            ex.shutdown(wait=True)
    records = list(records_by_id.values()) + [r for r in new_records if r is not None]
    if args.kanana and not args.dry_run and not stop.is_set():
        try:
            kanana_pass(records, out, log)
        except Exception as e:  # noqa: BLE001
            log.error("Kanana pass failed: %s: %s (drafts are intact; rerun with --resume --kanana)", type(e).__name__, str(e)[:200])
    merge_kanana(records, out)
    if not args.dry_run:
        try:
            write_workbook([r for r in records if r.get("status") not in ("error",)], out)
        except Exception as e:  # noqa: BLE001
            log.error("review.xlsx failed: %s: %s", type(e).__name__, str(e)[:200])
    try:
        import openai
        ffv = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        openai, ffv = None, None
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "args": {k: v for k, v in vars(args).items() if k != "env"},
                "guideline_version": GUIDELINE_VERSION, "prompt_id": PROMPT_ID, "prompt_hash": PROMPT_HASH, "index_sha256": index_hash, "git_commit": git_commit(),
                "versions": {"openai": getattr(openai, "__version__", None), "ffmpeg": ffv, "python": sys.version.split()[0]},
                "frame_schedule": "<10s:4 <60s:8 <300s:16 <900s:24 else 32 (superset of pipeline 3/5/8/12/16)", "selected": len(df), "processed": len(records),
                "status_counts": {s: sum(1 for r in records if r.get("status") == s) for s in ("ok", "needs_review", "unusable", "no_output", "unsupported_format", "error", "dry_run")},
                "cost_usd": round(sum(r.get("cost_usd", 0) for r in records), 4), "dry_run": args.dry_run, "stopped_early": stop.is_set()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("finished: %s", json.dumps(manifest["status_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
