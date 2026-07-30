#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, ssl, time, html, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

CTX = ssl._create_unverified_context()
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BASE = "https://www.kogl.or.kr"

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(1.0)
    return ""

def list_idxs(kw, page):
    q = urllib.parse.quote(kw)
    url = f"{BASE}/recommend/recommendList.do?division=img&searchStr={q}&koglCurrentPage={page}"
    h = fetch(url)
    seen, out = set(), []
    for m in re.finditer(r"recommendIdx=(\d+)", h):
        v = m.group(1)
        if v not in seen:
            seen.add(v); out.append(v)
    return out

DESC_RE = re.compile(r"desc-pop__body[^>]*>(.*?)</div>\s*</div>\s*</div>", re.DOTALL)
IMG_RE  = re.compile(r"(/upload_recommend/[^\"'()\s]+\.(?:jpg|png|jpeg|JPG|PNG|JPEG))")
CAT_RE  = re.compile(r"/upload_recommend/(?:thumb_[A-Z]/)?이미지/([^/]+)/")

def clean(t):
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    if t.endswith("닫기"):
        t = t[:-2].strip()
    return t

def detail(idx):
    url = f"{BASE}/recommend/recommendDivView.do?recommendIdx={idx}&division=img"
    h = fetch(url)
    if not h:
        return idx, "", "", ""
    dm = DESC_RE.search(h)
    desc = clean(dm.group(1)) if dm else ""
    imgs = IMG_RE.findall(h)
    img = ""
    if imgs:
        real = [x for x in imgs if "thumnaillist" in x.lower()]
        nonthumb = [x for x in imgs if "thumb" not in x.lower()]
        img = (real or nonthumb or imgs)[0]
    tm = re.search(r'property=["\']og:title["\']\s+content=["\'](.*?)["\']', h)
    title = clean(tm.group(1)) if tm else ""
    if not title:
        tt = re.search(r"<title>(.*?)</title>", h, re.DOTALL)
        title = clean(tt.group(1)) if tt else ""
    cat = ""
    cm = CAT_RE.search(img or h)
    if cm:
        cat = cm.group(1)
    return idx, desc, img, title, cat

KEYWORDS = ["전통","음식","한복","공예","농촌","항구","시장","스포츠","교통","산업","예술",
            "축제","마을","다리","등대","염전","도자기","자수","한지","서예","불교","무용",
            "방직","광부","어업","민속","놀이","세시","생활","노동","철도","우체국","소방",
            "농악","탈춤","베틀","대장간","옹기","한옥","서원","향교","누각","정원","수목원",
            "해녀","포구","광산","방앗간","장터",
            "근대건축","등록문화재","재실","종택","고택","사찰","석탑","읍성","산성","누정",
            "고분","민화","병풍","나전","청자","백자","분청","목가구","개항","의병",
            "근대","향약","비석","벽화","전수관","민속촌","서낭","당산","솟대","제방"]
MAX_PAGES = 3
POOL_CAP = 1300

def is_taxonomy(desc):
    return desc.count(" > ") >= 2 and bool(re.search(r"(동물계|원생생물계|식물계|균계|Animalia|Plantae|Protista)", desc))

def main():
    kw_used = list(KEYWORDS)
    tasks = [(kw, p) for kw in KEYWORDS for p in range(1, MAX_PAGES+1)]
    all_idx = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for (kw, p), ids in zip(tasks, ex.map(lambda t: list_idxs(*t), tasks)):
            for i in ids:
                all_idx.setdefault(i, kw)
    scanned = list(all_idx.keys())[:POOL_CAP]

    results = {}
    with ThreadPoolExecutor(max_workers=14) as ex:
        for tup in ex.map(detail, scanned):
            results[tup[0]] = tup[1:]

    # collect all qualifying, deduped
    qualifying = []   # (idx,len,title,img,cat,desc,taxonomy?)
    seen_desc = set()
    with_desc = 0
    for idx in scanned:
        desc, img, title, cat = results.get(idx, ("","","",""))
        if desc:
            with_desc += 1
        if not desc or desc == "-" or len(desc) < 80:
            continue
        sig = re.sub(r"[^0-9가-힣A-Za-z]", "", desc)[:70]
        if sig in seen_desc:
            continue
        seen_desc.add(sig)
        tax = is_taxonomy(desc)
        bucket = "TAXO" if tax else (cat or "기타")
        qualifying.append((idx, len(desc), title, img, bucket, desc))

    # diversity round-robin selection, cap TAXO at 3
    from collections import defaultdict, OrderedDict
    buckets = OrderedDict()
    for q in qualifying:
        buckets.setdefault(q[4], []).append(q)
    # sort each bucket by desc length desc (prefer richer)
    for b in buckets:
        buckets[b].sort(key=lambda x: -x[1])
    TARGET = 40
    taxo_cap = 3
    selected = []
    taxo_used = 0
    changed = True
    while len(selected) < TARGET and changed:
        changed = False
        for b, lst in buckets.items():
            if not lst:
                continue
            if b == "TAXO" and taxo_used >= taxo_cap:
                continue
            item = lst.pop(0)
            selected.append(item)
            if b == "TAXO":
                taxo_used += 1
            changed = True
            if len(selected) >= TARGET:
                break

    bucket_counts = defaultdict(int)
    for s in selected:
        bucket_counts[s[4]] += 1
    bc = ",".join(f"{k}:{v}" for k, v in sorted(bucket_counts.items(), key=lambda x:-x[1]))

    print(f"SUMMARY: scanned_idx_total={len(scanned)} detail_with_desc={with_desc} qualifying={len(qualifying)} accepted={len(selected)} buckets[{bc}] keywords={','.join(kw_used)}")
    for idx, ln, title, img, bucket, desc in selected:
        imgurl = (BASE + urllib.parse.quote(img)) if img else ""
        line = "\t".join([idx, str(ln),
                          (title or bucket).replace("\t"," ").replace("\n"," "),
                          imgurl,
                          desc[:150].replace("\t"," ").replace("\n"," ")])
        print(line)

if __name__ == "__main__":
    main()
