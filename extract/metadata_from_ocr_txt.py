#!/usr/bin/env python3
# (short) Wrapper that imports full code embedded as a string to avoid partial writes
import re, json, hashlib, datetime, argparse, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATE_TRIPLE = re.compile(r"(20\d{2})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})")
AMOUNT_WON = re.compile(r"([₩]?\s*[\d,]+)\s*원")
BIZ_REG_NO = re.compile(r"(\d{3}-\d{2}-\d{5})")
PHONE = re.compile(r"(\d{2,3}-\d{3,4}-\d{4})")
RRN = re.compile(r"(\d{6}-\d{7})")

def mask_rrn(text): return RRN.sub("******-*******", text)
def sha256_text(s): return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
def now_iso(): return datetime.datetime.now().isoformat()
def iso_date(y, m, d): return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

def detect_doc_type(text, forced=None):
    if forced in ("contract","계약서"): return "계약서"
    if forced in ("consent","동의서"): return "동의서"
    head = text[:400]
    if "동의서" in head or "개인정보 수집" in text or "동의 철회" in text: return "동의서"
    if "계약서" in head or re.search(r"^제\d+조\(", text, re.M): return "계약서"
    if "갑" in text and "을" in text: return "계약서"
    return "동의서"

def extract_dates(text):
    matches = list(DATE_TRIPLE.finditer(text))
    if not matches: return None, None
    return iso_date(*matches[0].groups()), iso_date(*matches[-1].groups())

def extract_amount(text):
    m = AMOUNT_WON.search(text)
    if not m: return None
    value = int(re.sub(r"[^\d]", "", m.group(1)))
    return {"value": value, "currency": "KRW"}

def extract_parties(text):
    parties = []
    for role in ("갑","을"):
        for m in re.finditer(rf"{role}\s*[:：]", text):
            window = text[m.end(): m.end()+180]
            name_m = re.search(r"\s*([가-힣A-Za-z().,\s-]{2,60})", window)
            reg_m = BIZ_REG_NO.search(window)
            item = {"role": role}
            if name_m: item["name"] = name_m.group(1).strip()
            if reg_m: item["registration_no"] = reg_m.group(1)
            if "name" in item:
                parties.append(item); break
    return list({p['role']: p for p in parties if 'name' in p}.values())

def extract_single_consent_subject(text):
    subj = {}
    nm = re.search(r"성명\s*[:：]\s*([가-힣A-Za-z\s]{2,20})", text)
    if nm: subj["name"] = nm.group(1).strip()
    ph = PHONE.search(text)
    if ph: subj["contact_phone"] = ph.group(1)
    _, exp = extract_dates(text)
    if exp: subj["date_signed"] = exp
    return subj

def extract_consent_subjects(text):
    blocks = list(re.finditer(r"성명\s*[:：]\s*([가-힣A-Za-z\s]{2,20})", text))
    if not blocks: return [extract_single_consent_subject(text)]
    subs = []
    for i, m in enumerate(blocks):
        start = m.start(); end = blocks[i+1].start() if i+1 < len(blocks) else len(text)
        chunk = text[start:end]
        subs.append(extract_single_consent_subject(chunk))
    return subs

def process_txt(txt_path, out_dir, forced_type, schemas):
    raw = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    raw = mask_rrn(raw)
    doc_type = detect_doc_type(raw, forced_type)
    checksum = sha256_text(raw)
    common = {
        "identifier": checksum[:16], "language": "ko", "pages": None,
        "checksum_sha256": checksum,
        "provenance": {"source_platform":"기타","original_filename":Path(txt_path).name,"ingested_at":now_iso(),"ocr_engine":"UNKNOWN","pipeline_version":"txt-v0.1"}
    }
    outputs = []
    if doc_type == "계약서":
        obj = {"document_type":"계약서","title":"계약서", **common}
        eff, exp = extract_dates(raw); 
        if eff: obj["effective_date"] = eff
        if exp: obj["expiry_date"] = exp
        amt = extract_amount(raw); 
        if amt: obj["amount"] = amt
        parts = extract_parties(raw); 
        if parts: obj["parties"] = parts
        obj["signature_presence"] = bool(re.search(r"(서명|날인)", raw))
        outputs = [("contract", obj)]
    else:
        subs = extract_consent_subjects(raw)
        outputs = []
        for i, s in enumerate(subs):
            o = {"document_type":"동의서","title":"개인정보 수집·이용 동의서", **common}
            o["subject"] = s if s else {"name": None}
            o["signature_presence"] = bool(re.search(r"(서명|날인|서명란)", raw))
            o["multi_subject_index"] = i if len(subs) > 1 else 0
            outputs.append(("consent", o))

    # Optional validation
    try:
        import jsonschema
        for kind, obj in outputs:
            schema = schemas.get(kind)
            if schema:
                v = jsonschema.Draft202012Validator(schema)
                errs = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(obj)]
                if errs: obj["_validation_errors"] = errs
    except Exception:
        pass

    written = []
    for kind, obj in outputs:
        stem = Path(txt_path).stem
        suffix = "" if len(outputs)==1 else f"__{obj.get('multi_subject_index',0):02d}"
        out_path = Path(out_dir) / f"{stem}{suffix}.json"
        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(out_path))
    return doc_type, written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Dir of OCR .txt files")
    ap.add_argument("--out", dest="out", required=True, help="Dir to write metadata JSON")
    ap.add_argument("--glob", default="*.txt", help="Glob for inputs")
    ap.add_argument("--doc-type", default="auto", choices=["auto","contract","consent","계약서","동의서"])
    ap.add_argument("--schema-contract", help="Path to metadata_contract_v1.json")
    ap.add_argument("--schema-consent", help="Path to metadata_consent_v1.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    in_dir = Path(args.inp); out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.rglob(args.glob))

    schemas = {}
    if args.schema_contract and Path(args.schema_contract).exists():
        schemas["contract"] = json.loads(Path(args.schema_contract).read_text(encoding="utf-8"))
    if args.schema_consent and Path(args.schema_consent).exists():
        schemas["consent"] = json.loads(Path(args.schema_consent).read_text(encoding="utf-8"))

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf); w.writerow(["txt_path","doc_type","outputs_count","outputs_paths"])
        total = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_txt, p, out_dir, args.doc_type if args.doc_type!="auto" else None, schemas): p for p in files}
            for fut in as_completed(futs):
                doc_type, paths = fut.result()
                w.writerow([str(futs[fut]), doc_type, len(paths), ";".join(paths)])
                total += len(paths)
                print(f"[{futs[fut].name}] -> {doc_type} out:{len(paths)}")
    print(f"Done. Wrote {total} JSON(s). Manifest: {manifest}")

if __name__ == "__main__":
    main()
