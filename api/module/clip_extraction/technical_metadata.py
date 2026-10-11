"""저작물 파일의 기술 속성 자동 추출 — 해상도 · 파일크기 · 파일 생성일.

연구개발계획서 2-4 '메타데이터 속성 추출'은 평가 대상 속성을 세 갈래로 규정한다:

    텍스트 속성(저작물정보) : 제목, 저자, 설명, 라이선스 유형, 키워드   → LLM/VLM 담당
    시각적 속성            : 해상도, 주요 색상, 개체 범주              → 해상도만 여기서
    파일정보 속성          : 파일크기, 파일포맷, 파일 생성 날짜          → 여기서

이 모듈이 담당하는 값은 전부 **결정적(deterministic)** 이다. 모델 호출이 없고,
환각이 원천적으로 불가능하며, 파일이 있는 한 채움률 100%다. 따라서 VLM에게
절대 추론시키지 않는다 — 영상 모델 비교시험(docs/영상모델_비교보고서_20260803.md
§4-1)에서 확인했듯 모델은 고유명사조차 창작하므로 기술 수치는 더 위험하다.

의존성은 전부 선택적이다. Pillow가 없으면 이미지 해상도만 비고, ffprobe가
없으면 영상 해상도만 빈다 — 파일크기·생성일은 표준 라이브러리만으로 항상 나온다.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".psd"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".wmv", ".mkv", ".webm", ".flv", ".mpg", ".mpeg",
              ".m4v", ".ts", ".swf"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma"}

# 확장자만 믿지 않는다. 보유 데이터셋에는 .swf 125건과 확장자 없는 파일이 섞여
# 있었고, 확장자 화이트리스트만 쓰면 이들이 통째로 미추출 처리된다.
# 아래 목록은 '미디어가 아님이 확실한' 확장자로, 여기 해당할 때만 탐침을 건너뛴다.
DOCUMENT_EXTS = {".pdf", ".hwp", ".hwpx", ".txt", ".doc", ".docx", ".ppt", ".pptx",
                 ".xls", ".xlsx", ".zip", ".json", ".xml", ".csv", ".md"}

_EXIF_DATETIME_ORIGINAL = 36867  # PIL ExifTags.TAGS 역참조 상수
_EXIF_DATETIME = 306


def detect_media_kind(file_path: str) -> str:
    """확장자로 미디어 종류 판별: image | video | audio | other."""
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "other"


def _to_date(value: Any) -> Optional[str]:
    """다양한 날짜 표기를 YYYY-MM-DD로 정규화. 실패하면 None."""
    if not value:
        return None
    s = str(value).strip()
    # EXIF는 "2019:08:13 14:22:01", 컨테이너 태그는 ISO8601
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:len(datetime.now().strftime(fmt))] if fmt.endswith("Z") else s,
                                     fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # 마지막 시도: 앞 10자가 날짜 꼴인지
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _image_properties(file_path: str) -> Dict[str, Any]:
    """Pillow로 이미지 해상도 + EXIF 촬영일시를 읽는다.

    ⚠️ Pillow는 약 1.79억 픽셀을 넘는 이미지에 DecompressionBombError를 던진다.
       공유마당 수집분에 2.58억 픽셀(고해상도 스캔본)이 실재한다. 치수만 읽는
       작업은 픽셀 버퍼를 만들지 않아 실제 DoS 위험이 없으므로, 이 함수 안에서만
       상한을 풀고 원복한다. 그래도 실패하면 ffprobe로 넘긴다 — 한 건의 예외가
       배치 전체를 죽이면 안 된다.
    """
    out: Dict[str, Any] = {}
    try:
        from PIL import Image
    except ImportError:
        return _video_properties(file_path)  # ffprobe가 이미지도 읽는다

    prev_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None            # 치수만 읽으므로 안전
    try:
        with Image.open(file_path) as im:
            out["width"], out["height"] = im.width, im.height
            try:
                exif = im._getexif()          # noqa: SLF001 — Pillow 공개 대안 없음
            except (AttributeError, OSError, TypeError, ValueError):
                exif = None
            if exif:
                raw = exif.get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
                d = _to_date(raw)
                if d:
                    out["created_date"] = d
                    out["created_date_source"] = "exif"
    except Exception:
        # PSD/손상 파일/미지원 포맷 등 — ffprobe로 재시도한다.
        out = {}
    finally:
        Image.MAX_IMAGE_PIXELS = prev_limit

    if not out.get("width"):
        fallback = _video_properties(file_path)
        if fallback.get("width"):
            fallback.pop("has_audio", None)
            fallback["dimension_source"] = "ffprobe(Pillow 실패)"
            return fallback
    return out


def _video_properties(file_path: str) -> Dict[str, Any]:
    """ffprobe로 영상/음성의 해상도·길이·코덱·오디오 유무·생성시각을 읽는다."""
    out: Dict[str, Any] = {}
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(file_path)],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return out
    if proc.returncode != 0 or not proc.stdout:
        return out
    try:
        probe = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return out

    streams = probe.get("streams") or []
    fmt = probe.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video:
        w, h = video.get("width"), video.get("height")
        if w and h:
            out["width"], out["height"] = w, h
            g = math.gcd(int(w), int(h))
            # display_aspect_ratio는 실측에서 15건 중 2건이 실제 픽셀비와 불일치했다.
            # 폭·높이의 최대공약수로 직접 약분한다.
            out["aspect_ratio"] = f"{int(w) // g}:{int(h) // g}"
        out["video_codec"] = video.get("codec_name")
        rate = video.get("r_frame_rate") or ""
        if "/" in rate:
            num, den = rate.split("/", 1)
            try:
                if float(den):
                    out["frame_rate"] = round(float(num) / float(den), 2)
            except ValueError:
                pass

    out["has_audio"] = audio is not None
    if audio:
        out["audio_codec"] = audio.get("codec_name")

    try:
        dur = float(fmt.get("duration") or 0)
        if dur > 0:
            out["duration_seconds"] = round(dur, 1)
            out["duration_display"] = f"{int(dur) // 3600:02d}:{int(dur) % 3600 // 60:02d}:{int(dur) % 60:02d}"
    except (TypeError, ValueError):
        pass

    d = _to_date((fmt.get("tags") or {}).get("creation_time"))
    if d:
        out["created_date"] = d
        out["created_date_source"] = "container"
    return out


def extract_technical_metadata(file_path: str) -> Dict[str, Any]:
    """저작물 파일에서 스키마 필드 + 부가 기술정보를 산출한다.

    Returns:
        {
          "resolution": "1920x1080" | None,     # 스키마 필드
          "file_size": 4629315 | None,          # 스키마 필드 (바이트)
          "file_created_date": "2013-10-12"|None,  # 스키마 필드
          "_technical": { ... }                 # 스키마 밖 부가정보
        }

        파일이 없으면 세 스키마 필드 모두 None이고 `_technical.error`가 채워진다.
    """
    result: Dict[str, Any] = {"resolution": None, "file_size": None, "file_created_date": None}
    tech: Dict[str, Any] = {}
    path = str(file_path) if file_path else ""

    if not path or not os.path.isfile(path):
        tech["error"] = "file not found"
        result["_technical"] = tech
        return result

    kind = detect_media_kind(path)

    # --- 파일크기: 항상 확보 가능 -----------------------------------------
    try:
        result["file_size"] = os.path.getsize(path)
    except OSError:
        pass

    # --- 미디어별 속성 ------------------------------------------------------
    props: Dict[str, Any] = {}
    if kind == "image":
        props = _image_properties(path)
    elif kind in ("video", "audio"):
        props = _video_properties(path)
    else:
        # 확장자로 판별 못한 파일 — 문서가 아니라면 실제로 열어 본다.
        # ffprobe → Pillow 순으로 시도하고, 성공한 쪽으로 종류를 확정한다.
        ext = os.path.splitext(path)[1].lower()
        if ext not in DOCUMENT_EXTS:
            props = _video_properties(path)
            if props.get("width") or props.get("duration_seconds"):
                kind = "video" if props.get("width") else "audio"
            else:
                props = _image_properties(path)
                if props.get("width"):
                    kind = "image"
            if kind != "other":
                tech["kind_detected_by"] = "content probe"

    tech["media_kind"] = kind

    if props.get("width") and props.get("height"):
        result["resolution"] = f"{props['width']}x{props['height']}"

    # --- 생성일: EXIF/컨테이너 태그 → 파일시스템 mtime 순 -------------------
    if props.get("created_date"):
        result["file_created_date"] = props["created_date"]
    else:
        try:
            # ctime은 리눅스에서 '생성'이 아니라 inode 변경 시각이라 쓰지 않는다.
            result["file_created_date"] = datetime.fromtimestamp(
                os.path.getmtime(path)).strftime("%Y-%m-%d")
            props["created_date_source"] = "filesystem_mtime"
        except (OSError, OverflowError, ValueError):
            pass

    tech.update(props)
    result["_technical"] = tech
    return result


if __name__ == "__main__":
    import sys

    targets = sys.argv[1:]
    if not targets:
        print("usage: python technical_metadata.py <file> [file ...]")
        raise SystemExit(1)
    for t in targets:
        r = extract_technical_metadata(t)
        print(f"\n▶ {os.path.basename(t)}")
        print(f"   resolution        : {r['resolution']}")
        print(f"   file_size         : {r['file_size']:,} bytes" if r["file_size"] else "   file_size         : None")
        print(f"   file_created_date : {r['file_created_date']}  ({r['_technical'].get('created_date_source')})")
        print(f"   _technical        : {json.dumps(r['_technical'], ensure_ascii=False)}")
