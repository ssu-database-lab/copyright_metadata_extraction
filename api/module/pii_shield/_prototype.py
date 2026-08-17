# -*- coding: utf-8 -*-
"""
pii_shield.py — PROTOTYPE: strip Korean PII (주민등록번호 등) from document
images / PDFs BEFORE anything is uploaded to an overseas OCR/LLM/VLM API.

Two paths:
  A) PDF WITH a real text layer  -> PyMuPDF true redaction (content removed, not covered)
  B) scanned image / image-only PDF -> local PaddleOCR (korean) gives per-line
     polygons -> regex + label-anchor rules -> pixels painted black -> re-encode.

Fail-closed design:
  - if the label "주민등록번호" is detected but no valid value line matched,
    the whole strip to the right of the label is blacked out anyway.
  - if the local OCR engine fails to load, the caller must abort (never
    silently pass the original to the cloud).
"""
from __future__ import annotations
import io, os, re, json, time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

# ---------------------------------------------------------------- patterns
# 주민등록번호: YYMMDD-SXXXXXX ; OCR often loses the hyphen or inserts spaces
RRN_STRICT = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})\s*[-–—~]\s*([1-8])(\d{6})(?!\d)")
RRN_NOHYPH = re.compile(r"(?<!\d)(\d{6})\s*([1-8]\d{6})(?!\d)")
# 외국인등록번호 shares the format (gender code 5-8)
FOREIGNER = RRN_STRICT
DRIVER   = re.compile(r"(?<!\d)\d{2}[-\s]?\d{2}[-\s]?\d{6}[-\s]?\d{2}(?!\d)")
PASSPORT = re.compile(r"\b[MSRODmsrod]\d{8}\b")
ACCOUNT  = re.compile(r"(?<!\d)\d{2,6}[-\s]\d{2,6}[-\s]\d{2,7}(?!\d)")
PHONE    = re.compile(r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}")
EMAIL    = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.\-]+")

LABELS_RRN = ("주민등록번호", "주민번호", "생년월일 및 주민", "외국인등록번호", "고유식별정보")

W = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)


def rrn_checksum_ok(d: str) -> bool:
    """d = 13 digits. Pre-2020.10 numbers satisfy this; newer ones may not."""
    if len(d) != 13 or not d.isdigit():
        return False
    s = sum(int(d[i]) * W[i] for i in range(12))
    return (11 - s % 11) % 10 == int(d[12])


def find_rrn_spans(text: str) -> List[Tuple[int, int, str, bool]]:
    """-> [(start, end, matched_text, checksum_valid)]"""
    out = []
    for rx in (RRN_STRICT, RRN_NOHYPH):
        for m in rx.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            out.append((m.start(), m.end(), m.group(0), rrn_checksum_ok(digits)))
    # de-dup overlapping
    out.sort()
    merged = []
    for s, e, t, ok in out:
        if merged and s < merged[-1][1]:
            continue
        merged.append((s, e, t, ok))
    return merged


# ---------------------------------------------------------------- audit
@dataclass
class Redaction:
    kind: str                 # RRN / RRN_LABEL_FALLBACK / PHONE / ...
    page: int
    bbox: Tuple[int, int, int, int]
    ocr_text: str = ""
    ocr_score: float = 0.0
    checksum_ok: Optional[bool] = None


@dataclass
class ShieldResult:
    ok: bool
    engine: str
    redactions: List[Redaction] = field(default_factory=list)
    elapsed_sec: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)


# ================================================================ PATH B
class ImageShield:
    """Local PaddleOCR -> polygons -> paint black. Nothing leaves the machine."""

    _ocr = None

    @classmethod
    def engine(cls, det="PP-OCRv5_mobile_det", rec="korean_PP-OCRv5_mobile_rec"):
        # GOTCHA: if you override text_detection_model_name you MUST also pin
        # text_recognition_model_name, otherwise paddleocr 3.7 silently falls
        # back to PP-OCRv6_medium_rec (CN/EN) and Korean comes out as garbage.
        if cls._ocr is None:
            from paddleocr import PaddleOCR
            cls._ocr = PaddleOCR(
                lang="korean",
                text_detection_model_name=det,
                text_recognition_model_name=rec,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
            )
        return cls._ocr

    @staticmethod
    def _poly_bbox(poly):
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def scrub(cls, img_path: str, out_path: str,
              also: Tuple[str, ...] = ("PHONE", "EMAIL", "ACCOUNT", "PASSPORT"),
              pad: int = 8) -> ShieldResult:
        from PIL import Image, ImageDraw
        t0 = time.time()
        ocr = cls.engine()
        res = ocr.predict(img_path)[0]
        texts, polys, scores = res["rec_texts"], res["rec_polys"], res["rec_scores"]

        im = Image.open(img_path).convert("RGB")
        drw = ImageDraw.Draw(im)
        reds: List[Redaction] = []

        extra = {"PHONE": PHONE, "EMAIL": EMAIL, "ACCOUNT": ACCOUNT,
                 "PASSPORT": PASSPORT, "DRIVER": DRIVER}

        rrn_line_idx = set()
        for i, tx in enumerate(texts):
            spans = find_rrn_spans(tx)
            if spans:
                x0, y0, x1, y1 = cls._poly_bbox(polys[i])
                drw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill="black")
                reds.append(Redaction("RRN", 0, (x0, y0, x1, y1), tx,
                                      round(float(scores[i]), 3), spans[0][3]))
                rrn_line_idx.add(i)
                continue
            for kind in also:
                if extra[kind].search(tx):
                    x0, y0, x1, y1 = cls._poly_bbox(polys[i])
                    drw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill="black")
                    reds.append(Redaction(kind, 0, (x0, y0, x1, y1), tx,
                                          round(float(scores[i]), 3)))
                    break

        # ---- fail-closed: label present but value not matched -> nuke the row
        Wimg, Himg = im.size
        for i, tx in enumerate(texts):
            if not any(lb in tx.replace(" ", "") for lb in LABELS_RRN):
                continue
            lx0, ly0, lx1, ly1 = cls._poly_bbox(polys[i])
            row_lo, row_hi = ly0 - 12, ly1 + 12
            covered = any(r.kind == "RRN" and not (r.bbox[3] < row_lo or r.bbox[1] > row_hi)
                          for r in reds)
            if covered:
                continue
            bx = (lx1 + 10, row_lo, Wimg - 40, row_hi)
            drw.rectangle(list(bx), fill="black")
            reds.append(Redaction("RRN_LABEL_FALLBACK", 0, bx, tx,
                                  round(float(scores[i]), 3)))

        im.save(out_path)
        return ShieldResult(True, "paddleocr:korean+PP-OCRv5_mobile_det",
                            reds, round(time.time() - t0, 2))


# ================================================================ PATH A
class PdfShield:
    """True redaction for PDFs. Removes the glyphs AND blanks image pixels."""

    @staticmethod
    def has_text_layer(doc, min_chars=40) -> bool:
        return any(len(p.get_text().strip()) >= min_chars for p in doc)

    @staticmethod
    def scrub(pdf_in: str, pdf_out: str, rasterize_fallback=True) -> ShieldResult:
        import fitz  # PyMuPDF
        t0 = time.time()
        doc = fitz.open(pdf_in)
        reds: List[Redaction] = []
        notes = []
        if not PdfShield.has_text_layer(doc):
            notes.append("no text layer -> image path required (rasterize + ImageShield)")
            return ShieldResult(False, "pymupdf", reds, round(time.time() - t0, 2), notes)

        for pno, page in enumerate(doc):
            words = page.get_text("words")          # (x0,y0,x1,y1, word, block, line, word_no)
            line_text, line_map = "", []
            for w in words:
                line_map.append((len(line_text), len(line_text) + len(w[4]), w))
                line_text += w[4] + " "
            for s, e, matched, ok in find_rrn_spans(line_text):
                for a, b, w in line_map:
                    if a < e and b > s:
                        r = fitz.Rect(w[0] - 1, w[1] - 1, w[2] + 1, w[3] + 1)
                        page.add_redact_annot(r, fill=(0, 0, 0))
                        reds.append(Redaction("RRN", pno,
                                              (int(r.x0), int(r.y0), int(r.x1), int(r.y1)),
                                              w[4], 1.0, ok))
            # PDF_REDACT_IMAGE_PIXELS(2) also blanks raster pixels under the rect
            page.apply_redactions(images=2, graphics=1, text=0)

        doc.save(pdf_out, garbage=4, deflate=True, clean=True)
        return ShieldResult(True, "pymupdf:true-redaction", reds,
                            round(time.time() - t0, 2), notes)


# ================================================================ verify
def verify_no_rrn_in_pdf(path: str) -> bool:
    import fitz
    doc = fitz.open(path)
    txt = "\n".join(p.get_text() for p in doc)
    return not find_rrn_spans(txt)


if __name__ == "__main__":
    import sys
    SCR = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "contract_scan_300dpi.jpg")
    dst = os.path.join(SCR, "shielded_" + os.path.basename(src).rsplit(".", 1)[0] + ".png")
    r = ImageShield.scrub(src, dst)
    print(r.to_json())
    print("wrote", dst)
