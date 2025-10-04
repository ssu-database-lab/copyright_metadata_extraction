import pymupdf  # 또는: import fitz as pymupdf
from pathlib import Path

src_root = Path("./document").resolve()            # Input Root
dst_root = Path("./converted_document").resolve()  # Output Root

def convert_tree(src: Path, dst: Path) -> None:
    for pdf_path in src.rglob("*.pdf"):
        rel = pdf_path.relative_to(src)         # A/B/sample.pdf
        out_dir = dst / rel.parent / pdf_path.stem  # .../A/B/sample/
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(pdf_path)
        try:
            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap()
                out_file = out_dir / f"{i:03}.png"   # 001.png, 002.png ...
                pix.save(out_file)
        finally:
            doc.close()

if __name__ == "__main__":
    convert_tree(src_root, dst_root)
