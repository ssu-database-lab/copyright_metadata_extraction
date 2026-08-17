"""영상 → 대표 키프레임 추출.

영상트랙 구현계획서 §3-4 의 선별 구조를 그대로 구현한다:

    목표의 1.75배 오버샘플 → 품질필터 → farthest-point(max-min) 다양성 선별 → N장

**farthest-point 선별이 핵심이다.** 앞에서부터 순차로 훑다가 목표 장수를 채우면
멈추는 탐욕적 방식은 뒤쪽 구간을 아예 보지 못한다 — 영상 모델 비교시험에서
5개 영상 전부가 앞 19~75% 구간만 관측된 원인이 그것이었다(비교보고서 §5).
전체 후보를 놓고 서로 가장 먼 프레임을 고르면 시간축에 고르게 퍼진다.

길이별 목표 프레임 수(계획서 §3-3): <10s 3 · 10~60s 5 · 1~5m 8 · 5~15m 12 · 15m+ 16.
연구 근거상 Qwen3-VL 계열은 8프레임 부근이 최적이고 16 이상은 이득이 없다.
우리 과업은 VQA가 아니라 메타데이터 추출이므로 벤치마크보다 적은 프레임으로 충분하다.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 품질 필터 임계값 (계획서 §3-4 실측)
_DARK_MEAN = 12          # 평균 휘도가 이보다 낮으면 암전으로 본다
_LOW_CONTRAST_RANGE = 20  # YMAX-YMIN 이 이보다 작으면 페이드/단색
_MIN_BYTES = 2000        # ffmpeg 이 만든 껍데기 JPEG 걸러내기


def target_frame_count(duration_s: float) -> int:
    """길이별 목표 프레임 수 (계획서 §3-3)."""
    if duration_s < 10:
        return 3
    if duration_s < 60:
        return 5
    if duration_s < 300:
        return 8
    if duration_s < 900:
        return 12
    return 16


def probe_duration(video_path: str) -> float:
    """ffprobe 로 재생시간(초). 실패하면 0."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return max(0.0, float(out))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, TypeError):
        return 0.0


def _dhash(path: str, size: int = 8) -> Optional[List[int]]:
    """8x8 difference hash — 시각적 유사도 비교용."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size))
            px = list(g.getdata())
    except Exception:
        return None
    bits = []
    for r in range(size):
        row = r * (size + 1)
        for c in range(size):
            bits.append(1 if px[row + c] > px[row + c + 1] else 0)
    return bits


def _hamming(a: List[int], b: List[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


def _quality_ok(path: str) -> bool:
    """암전·페이드·단색 프레임을 거른다.

    ⚠️ 평균 휘도만으로 자르면 실제 야간 씬(pblack≈60)까지 버린다. 그래서
    '거의 완전한 검정'과 '명암 범위가 거의 없음' 두 조건만 본다.
    """
    try:
        if os.path.getsize(path) < _MIN_BYTES:
            return False
        from PIL import Image
        with Image.open(path) as im:
            g = im.convert("L")
            lo, hi = g.getextrema()
            hist = g.histogram()
            total = sum(hist) or 1
            mean = sum(i * n for i, n in enumerate(hist)) / total
    except Exception:
        return False
    if mean < _DARK_MEAN:
        return False
    if (hi - lo) < _LOW_CONTRAST_RANGE:
        return False
    return True


def _grab(video_path: str, t: float, dst: str, long_edge: int = 768,
          duration: float = 0.0) -> bool:
    """t 초 지점 부근의 대표 프레임 1장을 뽑는다.

    -ss 를 -i 앞에 두어 키프레임 시크(빠름), 짧은 구간 안에서 `thumbnail` 이
    가장 대표성 있는 프레임을 고르게 한다.

    ⚠️ 창이 파일 끝을 넘어가면 ffmpeg 이 아무것도 내놓지 않는다. 3.7초 영상에서
       마지막 앵커가 그렇게 날아가 커버리지가 51%로 떨어졌다. 시크 지점과 창
       길이를 끝에 맞춰 줄인다.
    """
    seek = max(0.0, t)
    win = 2.0
    if duration > 0:
        seek = min(seek, max(0.0, duration - 0.35))
        win = max(0.35, min(2.0, duration - seek))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-ss", f"{seek:.2f}", "-t", f"{win:.2f}", "-i", str(video_path),
           "-vf", f"thumbnail=n={max(4, min(30, int(win * 15)))},"
                  f"scale={long_edge}:{long_edge}:force_original_aspect_ratio=decrease",
           "-frames:v", "1", "-q:v", "3", "-y", dst]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return os.path.exists(dst) and os.path.getsize(dst) > _MIN_BYTES


def _farthest_point_select(cands: List[Tuple[float, str, List[int]]],
                           k: int) -> List[Tuple[float, str, List[int]]]:
    """max-min 다양성 선별.

    시간축 양 끝을 먼저 확보한 뒤(처음·끝을 놓치면 영상의 시작과 결말을 못 본다),
    남은 자리는 '이미 고른 것들로부터 가장 먼' 후보로 채운다.
    """
    if k <= 0 or not cands:
        return []
    if len(cands) <= k:
        return list(cands)
    ordered = sorted(cands, key=lambda x: x[0])
    chosen = [ordered[0]]
    if k >= 2:
        chosen.append(ordered[-1])
    pool = [c for c in ordered if c not in chosen]
    while len(chosen) < k and pool:
        best, best_d = None, -1
        for c in pool:
            d = min(_hamming(c[2], s[2]) for s in chosen)
            if d > best_d:
                best, best_d = c, d
        chosen.append(best)
        pool.remove(best)
    return sorted(chosen, key=lambda x: x[0])


def _extract_without_duration(video_path: str, out: Path, target: int) -> Dict:
    """재생시간을 알 수 없는 컨테이너용 — 전체를 디코딩하며 대표 프레임 N장."""
    tmp = Path(tempfile.mkdtemp(prefix="kfnd_", dir=str(out)))
    try:
        pattern = str(tmp / "seq%03d.jpg")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
               "-i", str(video_path),
               "-vf", f"thumbnail=n=50,scale=768:768:force_original_aspect_ratio=decrease",
               "-frames:v", str(target), "-vsync", "vfr", "-q:v", "3", "-y", pattern]
        subprocess.run(cmd, capture_output=True, timeout=300)
        got = sorted(glob.glob(str(tmp / "seq*.jpg")))
        got = [g for g in got if _quality_ok(g)]
        if not got:
            return {"frames": [], "duration": 0.0, "target": target, "sampled": 0,
                    "kept": 0,
                    "error": "재생시간 미상 + 순차 디코딩에서도 프레임을 얻지 못했습니다"}
        frames = []
        for i, src in enumerate(got[:target], 1):
            dst = out / f"frame{i:02d}_seq.jpg"
            shutil.copy2(src, dst)
            frames.append({"path": str(dst), "t": float(i - 1)})   # 시점 미상 → 순번
        return {"frames": frames, "duration": 0.0, "target": target,
                "sampled": len(got), "kept": len(frames),
                "error": None, "note": "재생시간 미상 — 순차 디코딩으로 추출(시점은 순번)"}
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return {"frames": [], "duration": 0.0, "target": target, "sampled": 0,
                "kept": 0, "error": f"순차 디코딩 실패: {type(e).__name__}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def extract_keyframes(video_path: str, out_dir: str,
                      target: Optional[int] = None,
                      oversample: float = 1.75) -> Dict:
    """영상에서 대표 키프레임을 뽑아 out_dir 에 저장한다.

    Returns:
        {"frames": [{"path","t"}...], "duration": float, "target": int,
         "sampled": int, "kept": int, "error": str|None}
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video_path)
    if duration <= 0:
        # SWF(flv1) 처럼 컨테이너가 duration 을 노출하지 않는 경우가 있다.
        # 보유 데이터에 125건 존재하므로 통째로 버릴 수 없다 — 시크 대신 순차
        # 디코딩으로 대표 프레임을 뽑는다(파일이 작아 비용이 크지 않다).
        return _extract_without_duration(video_path, out, target or 5)

    tgt = target or target_frame_count(duration)
    n_over = max(tgt + 2, int(round(tgt * oversample)))
    # 앵커는 **전체 구간**에 균등 배치한다. 여기서 앞쪽에 몰면 뒤를 영영 못 본다.
    anchors = [duration * (i + 0.5) / n_over for i in range(n_over)]

    tmp = Path(tempfile.mkdtemp(prefix="kf_", dir=str(out)))
    cands: List[Tuple[float, str, List[int]]] = []
    try:
        for i, t in enumerate(anchors):
            raw = str(tmp / f"raw{i:02d}.jpg")
            if not _grab(video_path, t, raw, duration=duration):
                continue
            if not _quality_ok(raw):
                continue
            h = _dhash(raw)
            if h is None:
                continue
            cands.append((t, raw, h))

        if not cands:
            return {"frames": [], "duration": round(duration, 1), "target": tgt,
                    "sampled": len(anchors), "kept": 0,
                    "error": "품질 조건을 통과한 프레임이 없습니다 (전체 암전이거나 디코딩 실패)"}

        # 사전 중복제거는 하지 않는다. farthest-point 가 이미 '서로 가장 먼' 것을
        # 고르므로 근접 중복은 자연히 배제되고, 사전 제거는 오히려 시간축 끝단
        # 후보를 먼저 날려 커버리지를 떨어뜨린다(wmv 68%, mp4 51% 원인).
        picked = _farthest_point_select(cands, tgt)

        frames = []
        for i, (t, src, _) in enumerate(picked, 1):
            dst = out / f"frame{i:02d}_t{t:.1f}s.jpg"
            shutil.copy2(src, dst)
            frames.append({"path": str(dst), "t": round(t, 1)})
        return {"frames": frames, "duration": round(duration, 1), "target": tgt,
                "sampled": len(cands), "kept": len(frames), "error": None}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
