"""디렉터리/파일 조작 유틸."""
from pathlib import Path
from typing import List, Iterable

from module.parts.io import ensure_outdir, default_outfile  # noqa: F401


def iter_files_by_ext(root: Path, extensions: List[str]) -> Iterable[Path]:
    """지정 경로 내의 특정 확장자 파일들을 순회."""
    if not root.exists():
        return
    for ext in extensions:
        yield from root.rglob(f"*.{ext}")
        yield from root.rglob(f"*.{ext.upper()}")


def iter_document_files(root: Path) -> Iterable[Path]:
    """OCR 대상 문서 파일(pdf, img) 순회 (중복 제거)."""
    extensions = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    seen = set()
    for path in iter_files_by_ext(root, extensions):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield path
