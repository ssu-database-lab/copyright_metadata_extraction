"""
KOGL 상세페이지 GOLD 설명 + 대표 이미지 스크레이퍼 (익명 HTTP, 로그인 불필요).

배경: KOGL manifest(export)에는 저작물 설명이 없지만, KOGL '상세페이지'에는
`저작물 설명`(모달) 필드가 있고 관광/문화재 계열 작품은 이게 풍부하다(수원화성 2313자 등).
이 필드를 GOLD로, 대표 이미지를 입력으로 확보해 VLM(Gemma/Qwen) 설명 평가에 쓴다.

검증된 사실(2026-07-23 라이브):
  상세  : GET recommend/recommendDivView.do?recommendIdx=N&division=img (서버렌더, 익명)
  설명  : 모달 div  desc-pop__body[^>]*>(.*?)</div></div></div>  (값 '-' = 빈값)
  이미지: 첫 /upload_recommend/....(jpg|png) — https://www.kogl.or.kr + urlencode(path), 익명 ~700px
  ★ recommendIdx = 같은 설명을 공유하는 갤러리 → 대표 이미지 1장만 취함.

사용:
  python -m api.module.dataset_builder.kogl_gold_scraper --idx 37037,58758,57282 --out dataset/kogl_gold
  python -m api.module.dataset_builder.kogl_gold_scraper --idx-file idx.txt --min-desc 80 --out dataset/kogl_gold
  (idx-file: 한 줄에 recommendIdx 하나, 또는 TAB 구분 시 첫 컬럼을 idx로 사용)
"""
from __future__ import annotations
import argparse, html, re, ssl, time, urllib.parse, urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

BASE = "https://www.kogl.or.kr"
DETAIL = BASE + "/recommend/recommendDivView.do?recommendIdx={idx}&division={div}"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

_TAG = re.compile(r"<[^>]+>")
_DESC_MODAL = re.compile(r"desc-pop__body[^>]*>(.*?)</div>\s*</div>\s*</div>", re.S)
_DESC_DD = re.compile(r"<dt>\s*저작물 설명\s*</dt>\s*<dd>(.*?)</dd>", re.S)
_TITLE = re.compile(r"<(?:th|dt)>\s*저작물\s*명\s*</(?:th|dt)>\s*<(?:td|dd)[^>]*>(.*?)</(?:td|dd)>", re.S)
_IMG = re.compile(r'(/upload_recommend/[^"\']+?\.(?:jpg|jpeg|png|JPG|JPEG|PNG))')


def _clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = _TAG.sub(" ", html.unescape(s))
    s = "\n".join(ln.strip() for ln in s.splitlines() if ln.strip())
    return re.sub(r"[ \t]+", " ", s).strip()


def fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": BASE + "/"})
    return urllib.request.urlopen(req, context=_CTX, timeout=timeout).read()


def scrape_detail(idx: str, div: str = "img") -> dict:
    """상세페이지 → {title, gold_desc, image_url}."""
    text = fetch(DETAIL.format(idx=idx, div=div)).decode("utf-8", "ignore")
    m = _DESC_MODAL.search(text) or _DESC_DD.search(text)
    desc = _clean(m.group(1)) if m else ""
    if desc in ("-", ""):
        desc = ""
    mt = _TITLE.search(text)
    title = _clean(mt.group(1)) if mt else ""
    mi = _IMG.search(text)
    img = mi.group(1) if mi else ""
    return {"title": title, "gold_desc": desc, "image_url": img}


def download_image(image_path: str, dest: Path) -> Tuple[bool, int]:
    if not image_path:
        return False, 0
    url = BASE + urllib.parse.quote(image_path)
    try:
        data = fetch(url, timeout=60)
    except Exception:
        return False, 0
    if len(data) < 800 or not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        return False, len(data)
    dest.write_bytes(data)
    return True, len(data)


def _read_idx_file(path: Path) -> List[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split("\t")[0].strip()
        if first.isdigit():
            out.append(first)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="KOGL GOLD 설명 + 대표 이미지 스크레이퍼")
    ap.add_argument("--idx", default="", help="recommendIdx 쉼표구분")
    ap.add_argument("--idx-file", default="", help="recommendIdx 목록 파일(줄당 하나 / TAB 첫 컬럼)")
    ap.add_argument("--division", default="img")
    ap.add_argument("--out", default="dataset/kogl_gold")
    ap.add_argument("--min-desc", type=int, default=80, help="GOLD 설명 최소 길이(미만 제외)")
    ap.add_argument("--dedup-desc", action="store_true", help="동일 설명 중복 제거")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    idxs: List[str] = [x.strip() for x in args.idx.split(",") if x.strip().isdigit()]
    if args.idx_file:
        idxs += _read_idx_file(Path(args.idx_file))
    idxs = list(dict.fromkeys(idxs))
    if not idxs:
        print("recommendIdx가 없습니다 (--idx 또는 --idx-file).")
        return 2

    out = Path(args.out); img_dir = out / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_desc = set()
    print(f"대상 {len(idxs)}건 스크레이핑 (min-desc={args.min_desc}) ...")
    for i, idx in enumerate(idxs, 1):
        rec = {"recommendIdx": idx, "title": "", "gold_desc": "", "desc_len": 0,
               "image_file": "", "image_url": "", "status": ""}
        try:
            d = scrape_detail(idx, args.division)
        except Exception as e:
            rec["status"] = f"detail_fail:{type(e).__name__}"; records.append(rec)
            print(f"  ✗ [{i}/{len(idxs)}] {idx} {rec['status']}"); continue
        rec.update(title=d["title"], gold_desc=d["gold_desc"],
                   desc_len=len(d["gold_desc"]), image_url=d["image_url"])
        if rec["desc_len"] < args.min_desc:
            rec["status"] = "desc_short"; records.append(rec)
            print(f"  ∅ [{i}/{len(idxs)}] {idx} desc={rec['desc_len']}자 (<{args.min_desc})"); continue
        if args.dedup_desc:
            key = d["gold_desc"][:120]
            if key in seen_desc:
                rec["status"] = "dup_desc"; records.append(rec)
                print(f"  ⧗ [{i}/{len(idxs)}] {idx} 중복설명"); continue
            seen_desc.add(key)
        fname = f"{idx}.jpg"
        ok, size = download_image(d["image_url"], img_dir / fname)
        if ok:
            rec.update(image_file=fname, status="ok")
            print(f"  ✅ [{i}/{len(idxs)}] {idx} desc={rec['desc_len']}자 img={size//1024}KB {d['title'][:22]}")
        else:
            rec["status"] = "img_fail"
            print(f"  🚫 [{i}/{len(idxs)}] {idx} 이미지 실패")
        records.append(rec)
        time.sleep(args.sleep)

    # index
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "kogl_gold"
    cols = ["recommendIdx", "title", "gold_desc", "desc_len", "image_file", "image_url", "status"]
    ws.append(cols)
    for r in records:
        ws.append([r.get(c, "") for c in cols])
    wb.save(out / "kogl_gold.xlsx")
    from collections import Counter
    ok_n = sum(1 for r in records if r["status"] == "ok")
    print(f"\n=== 요약 === {dict(Counter(r['status'] for r in records))}")
    print(f"  채택(ok) {ok_n}건 | 이미지 {img_dir}/*.jpg | 인덱스 {out}/kogl_gold.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
