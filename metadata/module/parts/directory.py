import yaml
from pathlib import Path
from typing import Optional, List, Iterable

# ---------- directory ----------

def _iter_files(root: Path, extensions: List[str]) -> Iterable[Path]:
    """지정 경로 내의 특정 확장자 파일들을 순회."""
    if not root.exists():
        return []
    for ext in extensions:
        # 대소문자 모두 매칭 (Windows는 기본적으로 대소문자 무시하지만 명시적으로)
        yield from root.rglob(f"*.{ext}")
        yield from root.rglob(f"*.{ext.upper()}")

def _iter_text_files(root: Path) -> Iterable[Path]:
    """지정 경로 내의 .txt 파일만 순회."""
    return _iter_files(root, ["txt"])

def _iter_excel_files(root: Path) -> Iterable[Path]:
    """지정 경로 내의 Excel 파일(.xlsx, .xls) 순회."""
    return _iter_files(root, ["xlsx", "xls"])

def _iter_csv_files(root: Path) -> Iterable[Path]:
    """지정 경로 내의 .csv 파일만 순회."""
    return _iter_files(root, ["csv"])

def _iter_document_files(root: Path) -> Iterable[Path]:
    """OCR 대상 문서 파일(pdf, img) 순회."""
    # 지원 확장자 목록
    extensions = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    seen = set()
    for path in _iter_files(root, extensions):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield path

def _mirror_path(src_file: Path, src_root: Path, dest_root: Path, new_ext: str = ".txt") -> Path:
    """
    소스 파일의 경로 구조를 유지하며 대상 루트 아래의 경로를 생성합니다.
    """
    try:
        rel_path = src_file.relative_to(src_root)
    except ValueError:
        # src_file이 src_root의 하위가 아닌 경우 (드물지만 처리)
        rel_path = Path(src_file.name)
        
    dest_path = dest_root / rel_path.with_suffix(new_ext)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    return dest_path

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


def iter_text_files(root: Path) -> Iterable[Path]:
    """지정 경로 내의 .txt 파일만 순회하는 이터레이터 반환."""
    return _iter_text_files(root)

def iter_excel_files(root: Path) -> Iterable[Path]:
    """지정 경로 내의 Excel 파일만 순회하는 이터레이터 반환."""
    return _iter_excel_files(root)

def iter_csv_files(root: Path) -> Iterable[Path]:
    """지정 경로 내의 .csv 파일만 순회하는 이터레이터 반환."""
    return _iter_csv_files(root)

def iter_document_files(root: Path) -> Iterable[Path]:
    """OCR 대상 문서 파일만 순회하는 이터레이터 반환."""
    return _iter_document_files(root)

def get_mirror_output_path(src_file: Path, src_root: Path, dest_root: Path) -> Path:
    """입력 파일의 상대 경로를 유지하여 출력 경로 반환."""
    return _mirror_path(src_file, src_root, dest_root)
