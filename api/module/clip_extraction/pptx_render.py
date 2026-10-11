"""PPTX 1쪽 슬라이드 → PNG 렌더링 (PowerPoint COM 을 powershell.exe 로 호출) — M2 spec A7.

공유마당의 PPT 템플릿 저작물(예: 13385261 'PPT 템플릿_여행계획용_001')은 사이트가 '이미지'로
분류하지만 파일은 .pptx 다. VLM 은 그림만 읽으므로 1쪽을 PNG 로 구워 넣는다.
dataset_builder/generate_descriptions.render_pptx_first_slide 와 같은 방식(PowerPoint COM)이지만,
파이프라인용으로 임시 경로를 하드코딩하지 않고(%TEMP%\\pptx_render 또는 PPTX_RENDER_TMP),
원본을 윈도 쪽 임시 폴더로 복사해 렌더링한다 — COM 은 \\\\wsl.localhost\\… 경로를 믿을 수 없다.

렌더링이 불가능한 환경(리눅스 서버, PowerPoint 미설치)에서는 PptxRenderError 를 던진다.
호출자(pipeline.run_vlm)는 이를 잡아 success=False 와 명확한 error 문자열로 응답한다 — 죽지 않는다.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PptxRenderError(RuntimeError):
    """렌더링 불가 또는 실패. 메시지에 원인을 담는다."""


_PS1 = r"""param([string]$In, [string]$Out, [int]$W, [int]$H)
$app = New-Object -ComObject PowerPoint.Application
try {
  $pres = $app.Presentations.Open($In, $true, $false, $false)
  try { $pres.Slides.Item(1).Export($Out, "PNG", $W, $H); "OK" } finally { $pres.Close() }
} finally { $app.Quit() }
"""

_FALLBACK_POWERSHELL = (
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe",
)


def powershell_path() -> Optional[str]:
    """powershell.exe 경로 — PATH 에 없으면(systemd 서비스 등) 기본 설치 위치를 본다. 없으면 None."""
    found = shutil.which("powershell.exe")
    if found:
        return found
    for cand in _FALLBACK_POWERSHELL:
        if os.path.exists(cand):
            return cand
    return None


def is_available() -> bool:
    return powershell_path() is not None


def _to_windows(p: Path) -> str:
    try:
        out = subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return str(p)
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else str(p)


def _to_wsl(win_path: str) -> Optional[Path]:
    try:
        out = subprocess.run(["wslpath", "-u", win_path], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    s = out.stdout.strip()
    return Path(s) if out.returncode == 0 and s else None


def _work_dir(ps: str) -> Path:
    """PowerPoint(윈도 프로세스)가 읽고 쓸 수 있는 임시 폴더.

    PPTX_RENDER_TMP 가 있으면 그것(WSL 경로), 없으면 %TEMP%\\pptx_render 를 wslpath 로 바꿔 쓴다.
    """
    env = os.getenv("PPTX_RENDER_TMP")
    if env:
        d = Path(env)
    else:
        try:
            r = subprocess.run([ps, "-NoProfile", "-Command", "[System.IO.Path]::GetTempPath()"],
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            raise PptxRenderError(f"powershell.exe 실행 실패: {type(e).__name__}: {e}") from e
        lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        win_tmp = lines[-1] if lines else ""
        wsl_tmp = _to_wsl(win_tmp) if win_tmp else None
        if wsl_tmp is None:
            raise PptxRenderError("윈도 임시 폴더(%TEMP%)를 찾지 못함 — PPTX_RENDER_TMP 로 지정하세요")
        d = wsl_tmp / "pptx_render"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_first_slide(src, dst_png, width: int = 1920, height: int = 1080,
                       timeout: int = 180) -> Path:
    """src(.pptx)의 1쪽 슬라이드를 dst_png 로 렌더링하고 그 경로를 돌려준다. 실패하면 PptxRenderError."""
    src = Path(src)
    dst = Path(dst_png)
    if not src.is_file():
        raise PptxRenderError(f"원본 파일 없음: {src}")
    ps = powershell_path()
    if ps is None:
        raise PptxRenderError(
            "powershell.exe 를 찾을 수 없음 — .pptx 렌더링은 PowerPoint 가 설치된 Windows/WSL2 에서만 가능")

    work = _work_dir(ps)
    token = uuid.uuid4().hex[:12]
    tmp_src = work / f"{token}.pptx"        # ASCII 이름 — 한글 경로·공백을 COM 인자에서 피한다
    tmp_png = work / f"{token}.png"
    ps1 = work / f"render_{token}.ps1"
    try:
        shutil.copyfile(src, tmp_src)
        ps1.write_bytes(b"\xef\xbb\xbf" + _PS1.encode("utf-8"))   # BOM 이 있어야 PS 가 UTF-8 로 읽는다
        cmd = [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", _to_windows(ps1),
               "-In", _to_windows(tmp_src), "-Out", _to_windows(tmp_png),
               "-W", str(width), "-H", str(height)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise PptxRenderError(f"PowerPoint 렌더링 시간 초과({timeout}s): {src.name}") from e
        except OSError as e:
            raise PptxRenderError(f"powershell.exe 실행 실패: {e}") from e
        if not tmp_png.exists() or tmp_png.stat().st_size == 0:
            tail = ((r.stdout or "")[-200:] + " " + (r.stderr or "")[-200:]).strip()
            raise PptxRenderError(f"PowerPoint 렌더링 실패: {src.name} — {tail or 'PNG 가 생성되지 않음'}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_png, dst)
        logger.info("PPTX 1쪽 렌더링 완료: %s → %s (%d bytes)", src.name, dst.name, dst.stat().st_size)
        return dst
    finally:
        for p in (tmp_src, tmp_png, ps1):
            try:
                p.unlink()
            except OSError:
                pass
