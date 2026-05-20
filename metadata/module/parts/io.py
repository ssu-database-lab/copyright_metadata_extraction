"""파일 I/O 유틸 — 출력 디렉터리 생성, 기본 출력 경로 계산."""
from pathlib import Path
from typing import Optional


def ensure_outdir(out_dir: str = "data/out/results") -> Path:
    """출력 디렉터리를 생성하고 Path 객체를 반환."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_outfile(file_path: Optional[str], out_dir: Path) -> Path:
    """파일 경로를 기반으로 기본 출력 파일 Path를 생성."""
    name = Path(file_path).stem if file_path else "untitled"
    return out_dir / f"{name}_metadata.json"
