import yaml
from pathlib import Path
from typing import Optional, List

# ---------- directory ----------

def _load_schema_labels() -> List[str]:
    """labels.yaml에서 schema_labels 불러오기"""
    with open("configs/labels.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["schema_labels"]

def _ensure_outdir(out_dir: str = "data/out/results") -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def _default_outfile(file_path: Optional[str], out_dir: Path) -> Path:
    if file_path:
        name = Path(file_path).stem
    else:
        name = "untitled"
    return out_dir / f"{name}_metadata.json"


# ---------- public helpers ----------

def load_schema_labels() -> List[str]:
    """외부 모듈에서 사용하는 공개 함수."""
    return _load_schema_labels()


def ensure_outdir(out_dir: str = "data/out/results") -> Path:
    """출력 디렉터리를 생성하고 Path 객체를 반환."""
    return _ensure_outdir(out_dir)


def default_outfile(file_path: Optional[str], out_dir: Path) -> Path:
    """파일 경로를 기반으로 기본 출력 파일 Path 생성."""
    return _default_outfile(file_path, out_dir)