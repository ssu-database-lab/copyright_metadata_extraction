import yaml
from pathlib import Path
from typing import Optional, List, Iterable

# ---------- 파일 순회 ----------

def iter_files_by_ext(root: Path, extensions: List[str]) -> Iterable[Path]:
    """지정 경로 내의 특정 확장자 파일들을 순회."""
    if not root.exists():
        return
    for ext in extensions:
        yield from root.rglob(f"*.{ext}")
        yield from root.rglob(f"*.{ext.upper()}")


# ---------- 스키마 라벨 로드 ----------

def load_schema_labels() -> List[str]:
    """labels.yaml에서 schema_labels 불러오기."""
    with open("configs/labels.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["schema_labels"]


def iter_document_files(root: Path) -> Iterable[Path]:
    """OCR 대상 문서 파일(pdf, img) 순회 (중복 제거)."""
    extensions = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    seen = set()
    for path in iter_files_by_ext(root, extensions):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield path


# 경로 처리 함수
def get_mirror_output_path(src_file: Path, src_root: Path, dest_root: Path, new_ext: str = ".txt") -> Path:
    """소스 파일의 경로 구조를 유지하며 대상 루트 아래의 경로를 생성."""
    try:
        rel_path = src_file.relative_to(src_root)
    except ValueError:
        rel_path = Path(src_file.name)
    
    dest_path = dest_root / rel_path.with_suffix(new_ext)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    return dest_path


# 출력 디렉터리 생성 함수
def ensure_outdir(out_dir: str = "data/out/results") -> Path:
    """출력 디렉터리를 생성하고 Path 객체를 반환."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# 기본 출력 파일 경로 생성 함수
def default_outfile(file_path: Optional[str], out_dir: Path) -> Path:
    """파일 경로를 기반으로 기본 출력 파일 Path 생성."""
    name = Path(file_path).stem if file_path else "untitled"
    return out_dir / f"{name}_metadata.json"
