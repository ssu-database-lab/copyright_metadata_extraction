"""프로젝트 루트 기준 경로 해석."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_user_path(path_str: str) -> Path:
    """프로젝트 루트 기준 상대 경로 해석."""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


def resolve_out_dir(
    output_path: str,
    model_display: Optional[str],
    result_phase: Optional[str] = None,
) -> str:
    """output_path/[model_display]/[result_phase]/ 경로 문자열."""
    base = output_path
    if model_display is not None:
        base = str(Path(output_path) / model_display)
    if result_phase:
        base = str(Path(base) / result_phase)
    return base
