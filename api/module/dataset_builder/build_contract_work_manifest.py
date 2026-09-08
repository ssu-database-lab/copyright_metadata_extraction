"""계약서 ↔ 저작물 매칭 매니페스트 생성기.

생성계약서.zip 의 계약서 PDF 5,834건을 각자의 저작물(이미지/영상/어문)과 짝지어
dataset/contract_work_manifest.csv 를 만든다. 이 CSV 로 /v3 평가용 ZIP 을 만든다.

조인 키는 **파일명 앞의 숫자 ID**. 계약서 본문에는 저작물 ID 가 인쇄되지 않으므로
(비어있지 않은 5,714건 전수 확인, 출현 0건) 파일명이 유일한 연결고리다. 대신 제목·저작자·
권리자·이용허락기간은 본문에 그대로 인쇄되어 있어 매칭을 내용으로 검증할 수 있다
(14개 층을 모두 덮는 42건 층화표본에서 제목·날짜 전건 일치, 권리자 5,714/5,714).

알려진 결함 — CSV 의 exclude_reason 으로 표시된다:
  · 계약서 PDF 118건이 0바이트 (원본 ZIP 안에서 이미 0바이트 — hwpx→pdf 변환 실패)
  · 저작물 없음 59건 (KOGL 영상 57 + 어문 2). 어문 2건은 KOGL 서버에서 원본 삭제 — 복구 불가
  · dataset/{images,videos,documents} 는 330x230 썸네일이라 저작물로 쓰지 않는다

사용:  python -m module.dataset_builder.build_contract_work_manifest
"""

from __future__ import annotations

import argparse, hashlib, json, os, re, subprocess, sys
from collections import Counter, defaultdict

import pandas as pd

ROOT   = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
PDFDIR = f"{ROOT}/dataset/생성계약서_pdf"
ZIP    = f"{ROOT}/dataset/생성계약서.zip"
XLSX   = f"{ROOT}/dataset/생성계약서_메타데이터_항목.xlsx"  # ZIP 내 "1.메타데이터 항목.xlsx"
OUT    = f"{ROOT}/dataset/contract_work_manifest.csv"


MEDIA={"어문":"text","이미지":"image","영상":"video"}
JONGBYEOL={"어문":"어문저작물","이미지":"사진저작물","영상":"영상저작물"}   # 계약서에 인쇄되는 종별
BUCKET={"만료":"expired","기증":"donated","CCL":"ccl","KOGL":"kogl","만료-KOGL":"expired_kogl"}
# router.py 확장자 → modality (document→text). psd/swf 는 라우팅은 되지만 디코딩이 안 된다.
IMAGE_EXTS={"jpg","jpeg","png","gif","bmp","tiff","tif","webp","psd","heic"}
VIDEO_EXTS={"mp4","avi","mov","mkv","webm","wmv","swf","mpg","mpeg","m4v","ts","flv","3gp","ogv"}
DOC_EXTS={"pdf","hwp","docx","doc","pptx","xlsx"}; TEXT_EXTS={"txt","md","ocr"}
# 확장자만 보고 "처리 불가"로 단정하지 않는다. .swf 9건과 무확장자 1건을 ffmpeg 로
# 실제 디코딩해 본 결과 10건 모두 768x432 키프레임이 정상 추출됐다(swf=flv1+mp3,
# 무확장자=Microsoft ASF wmv2). 따라서 실측으로 확인된 것만 배제한다.
UNDECODABLE={".psd"}
PROBE_EXTS={""}          # 확장자가 없으면 ffprobe 로 컨테이너를 알아낸다
# .hwp 는 universal_ocr.extract_text() 가 pyhwp 로 본문을 직접 읽는다(래스터화 없음).
# 실측: 저작물 .hwp 40건 중 39건이 157~33,431자로 추출되고, 나머지 1건은 확장자만
# .HWP 인 PDF 라 _process_hwp 가 PDF 경로로 넘긴다. 표만 남는 문서는 추출기가
# None 을 돌려 OCR 경로로 넘어가므로 여기서 미리 배제할 이유가 없다.
UNPROCESSABLE={".zip",".pptx",".ppt",".xlsx",".xls",".hwpx"}
THUMBS={"local_preview"}                            # 330x230 플레이스홀더/축소본 — 저작물 아님
SRC_RANK={"gongu_gt":0,"kogl_originals":1,"gongu_fs":2}
ZIP_CAP={"image":1_500_000,"text":1_500_000,"video":3_000_000}
# 감사에서 확인된 개별 결함
GLYPH_BAD={"13315744","13315759","13315736","13315725","13266571","13266660"}
DL_FAILED={"143943","144247"}                       # KOGL 서버에서 원본 삭제됨 — 재시도 불가
NULLISH={"-","-, -","소속없음","", "nan"}

def probe_container(path):
    """확장자가 없는 파일의 실제 컨테이너를 알아낸다. 실패하면 None."""
    try:
        r=subprocess.run(["ffprobe","-v","error","-show_entries","format=format_name",
                          "-of","default=nw=1:nk=1",path],
                         capture_output=True,text=True,timeout=60)
        fmt=(r.stdout or "").strip().split(",")[0]
        return {"asf":".wmv","matroska":".mkv","mov":".mp4","mpegts":".ts"}.get(fmt, f".{fmt}" if fmt else None)
    except Exception:
        return None


def modality(ext):
    e=ext.lower().lstrip(".")
    if e in IMAGE_EXTS: return "image"
    if e in VIDEO_EXTS: return "video"
    if e in DOC_EXTS or e in TEXT_EXTS: return "text"
    return "unknown"



def build_work_index():
    """공유마당 GT · E: 파일스캔 · KOGL 원본 · 로컬 썸네일을 저작물 ID 로 색인한다."""
    idx = defaultdict(list)   # id -> [ {source, path, media, bucket, title, size} ]

    # ---- 1) 공유마당 ground_truth (authoritative: media+bucket+title) ----
    gt_meta = {}
    for m in ("image","video","text"):
        for b in ("expired","donated","ccl","kogl"):
            p=f"/mnt/d/copyright_dataset_metadata/{m}/{b}/ground_truth.jsonl"
            if not os.path.isfile(p): continue
            for line in open(p,encoding="utf-8"):
                try: g=json.loads(line)
                except Exception: continue
                i=str(g["id"]); f=g.get("file","")
                gt_meta[i]={"media":g["media"],"bucket":g["license_bucket"],
                            "title":(g.get("attributes",{}).get("제목") or {}).get("value"),
                            "gt_file":f,"file_exists":bool(g.get("file_exists"))}
                if f: idx[i].append({"source":"gongu_gt","path":f,"media":m,"bucket":b,
                                     "size":os.path.getsize(f) if os.path.isfile(f) else 0})

    # ---- 2) raw filesystem scan of E:/gongu_dataset (catches files absent from GT) ----
    for m in ("image","video","text"):
        for b in ("expired","donated","ccl","kogl"):
            d=f"/mnt/e/gongu_dataset/{m}/{b}"
            if not os.path.isdir(d): continue
            for fn in os.listdir(d):
                fp=os.path.join(d,fn)
                if not os.path.isfile(fp): continue
                if fn.endswith((".jsonl",".xlsx",".part")): continue
                mm=re.match(r'^(\d+)_', fn)
                if not mm: continue
                i=mm.group(1)
                if any(x["path"]==fp for x in idx[i]): continue
                idx[i].append({"source":"gongu_fs","path":fp,"media":m,"bucket":b,
                               "size":os.path.getsize(fp)})

    # ---- 3) KOGL originals (E:) ----
    ko=pd.read_excel("/mnt/e/kogl_originals/originals_index.xlsx")
    KMEDIA={"이미지":"image","영상":"video","어문":"text"}
    kogl_meta={}
    for _,r in ko.iterrows():
        i=str(r["원문인덱스"]); sub=str(r["분류"])
        kogl_meta[i]={"title":r.get("제목"),"media":KMEDIA.get(sub,sub),"status":r.get("status")}
        sf=r.get("saved_file")
        if isinstance(sf,str) and sf:
            fp=f"/mnt/e/kogl_originals/{sub}/{sf}"
            if os.path.isfile(fp):
                idx[i].append({"source":"kogl_originals","path":fp,"media":KMEDIA.get(sub,sub),
                               "bucket":"kogl","size":os.path.getsize(fp)})

    # ---- 4) local dataset previews (KOGL png/jpg renders) ----
    for sub,media in (("images","image"),("videos","video"),("documents","text")):
        d=f"/home/mbmk92/copyright/copyright_metadata_extraction/dataset/{sub}"
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            fp=os.path.join(d,fn)
            if not os.path.isfile(fp): continue
            i=os.path.splitext(fn)[0]
            if not i.isdigit(): continue
            idx[i].append({"source":"local_preview","path":fp,"media":media,
                           "bucket":None,"size":os.path.getsize(fp)})

    return idx, gt_meta, kogl_meta


def build_manifest(idx, gt_meta, kogl_meta, out=OUT):
    """계약서 ID 로 조인해 매니페스트 CSV 를 쓴다."""
    # ---------- 계약서 ----------
    pdf_by_id={}; noid=[]
    for f in os.listdir(PDFDIR):
        m=re.match(r'^(\d+)\s', f)
        pdf_by_id.setdefault(m.group(1), f) if m else noid.append(f)
    # PDF 파일명은 xlsx 파일명을 마지막 '.' 에서 자르고 ':' 등 윈도우 금지문자를 치환한 것.
    # '/' 가 들어간 2건은 앞부분이 통째로 잘려 ID 를 잃었다 — 제목으로 복원한다.
    SLASH_FIX={"6학년용.pdf":"144066","AR 산업 활성화를 위한 법정책적 과제.pdf":"64823"}
    for f in noid:
        if f in SLASH_FIX: pdf_by_id[SLASH_FIX[f]]=f
    hwpx_by_id={}
    for n in subprocess.run(["unzip","-Z1",ZIP],capture_output=True,text=True).stdout.splitlines():
        if n.endswith(".hwpx"):
            m=re.match(r'^(\d+)\s', n)
            if m: hwpx_by_id[m.group(1)]=n

    x=pd.read_excel(XLSX,"데이터 랜덤 값")
    x["wid"]=x["파일명"].astype(str).str.extract(r'^(\d+)\s')[0]
    x["wtitle"]=x["파일명"].astype(str).str.replace(r'^\d+\s','',regex=True)

    # ---------- 저작물 선택 ----------
    def pick(wid, want):
        real=[e for e in idx.get(wid,[]) if e["source"] not in THUMBS]
        thumb=[e for e in idx.get(wid,[]) if e["source"] in THUMBS]
        if not real:
            return None, [], ("placeholder_only" if thumb else "no_work_file")
        def score(e):
            ext=os.path.splitext(e["path"])[1].lower()
            return (0 if modality(ext)==want else 1,          # 1순위: 선언 미디어와 일치
                    0 if ext not in UNPROCESSABLE else 1,      # 2순위: 파이프라인이 여는 형식
                    0 if ext not in UNDECODABLE else 1,        # 3순위: 실제 디코딩 가능
                    SRC_RANK.get(e["source"],9), -e["size"])
        ranked=sorted(real,key=score)
        return ranked[0], ranked[1:], ""

    rows=[]
    for _,r in x.iterrows():
        wid=str(r["wid"]); declared=MEDIA.get(r["저작물형태"]); braw=str(r["공유형태"])
        pdf=pdf_by_id.get(wid)
        ppath=f"{PDFDIR}/{pdf}" if pdf else ""
        pbytes=os.path.getsize(ppath) if ppath and os.path.isfile(ppath) else 0
        best,alts,why=pick(wid,declared)

        ext=os.path.splitext(best["path"])[1].lower() if best else ""
        probed=""
        if best and ext in PROBE_EXTS:
            # 확장자가 없으면 라우터가 unknown 으로 흘려보낸다. 실제 컨테이너를 읽어
            # works/{id}.wmv 처럼 담아야 영상 경로를 탄다.
            probed=probe_container(best["path"]) or ""
            if probed: ext=probed
        actual=modality(ext) if best else ""
        # 두 축을 분리한다.
        #   blocking  = 파이프라인이 아예 못 돈다 (계약서/저작물 파일 자체 문제)
        #   scoring   = 돌긴 도는데 특정 속성의 정답을 신뢰할 수 없다
        blocking=[]; scoring=[]
        if not pdf:            blocking.append("CONTRACT_PDF_MISSING")
        elif pbytes==0:        blocking.append("CONTRACT_PDF_EMPTY")
        if best is None:       blocking.append("WORK_"+why.upper())
        elif ext in UNPROCESSABLE: blocking.append("WORK_EXT_UNPROCESSABLE")
        elif ext in UNDECODABLE:   blocking.append("WORK_EXT_UNDECODABLE")
        # 미디어 라벨이 어긋나도 라우터는 확장자로 분기하므로 처리 자체는 된다.
        # (어문 저작물이 .jpg 악보 스캔인 경우 등 — 출처 카탈로그의 라벨 오류)
        if best is not None and actual!=declared: scoring.append("MEDIA_RELABELED")
        # 제목의 악센트 글리프가 PDF 폰트(한양신명조)에 없어 두부 상자로 찍혔다.
        # 본문 나머지는 정상이므로 제목만 채점에서 빼면 된다.
        if wid in GLYPH_BAD:   scoring.append("TITLE_GLYPH_CORRUPT")
        reasons=blocking+scoring

        status = ("OK" if not reasons else
                  "BOTH_MISSING" if (pbytes==0 and best is None) else
                  "NO_WORK" if best is None else
                  "NO_CONTRACT_PDF" if pbytes==0 else
                  "UNUSABLE_WORK" if blocking else "OK_WITH_WARNING")
        gm=gt_meta.get(wid,{}); km=kogl_meta.get(wid,{})
        author=str(r.get("저작자") or "").strip()
        rows.append({
          "set_id":wid, "status":status,
          "eval_ready": not blocking,          # 파이프라인 실행 가능
          "gt_title_scorable": wid not in GLYPH_BAD,
          "exclude_reason": ";".join(reasons),
          # --- 계약서 ---
          "contract_pdf": os.path.relpath(ppath,ROOT) if ppath else "",
          "contract_pdf_bytes": pbytes,
          "contract_hwpx_in_zip": hwpx_by_id.get(wid,""),
          "document_type": "저작재산권 이용허락 계약서",
          # --- 저작물 ---
          "work_path": best["path"] if best else "",
          "work_arcname": f"works/{wid}{ext}" if best else "",
          "work_ext": ext, "work_ext_probed": bool(probed), "work_bytes": best["size"] if best else 0,
          "work_source": best["source"] if best else "",
          "work_candidate_count": len(idx.get(wid,[])),
          "work_alt_paths": "|".join(e["path"] for e in alts),
          "work_download_status": "failed_file_missing" if wid in DL_FAILED else "",
          "media": actual, "media_declared": declared,
          "fits_zip_cap": bool(best and best["size"]<=ZIP_CAP.get(declared,1_500_000)),
          # --- 라이선스 ---
          "license_bucket": BUCKET.get(braw,braw), "license_bucket_raw": braw,
          # --- 계약서에 실제로 인쇄되는 정답값 (xlsx = PDF 본문, 전수 검증됨) ---
          "gt_title": r["wtitle"],
          "gt_author": "" if author in NULLISH else author,
          "gt_rights_holder": r.get("권리자"),
          "gt_licensee": r.get("이용자"),
          "gt_work_kind": JONGBYEOL.get(r["저작물형태"],""),
          "gt_license_start": f"{r['이용허락기간_SY']:.0f}-{r['이용허락기간_SM']:02.0f}-{r['이용허락기간_SD']:02.0f}" if pd.notna(r.get("이용허락기간_SY")) else "",
          "gt_license_end":   f"{r['이용허락기간_EY']:.0f}-{r['이용허락기간_EM']:02.0f}-{r['이용허락기간_ED']:02.0f}" if pd.notna(r.get("이용허락기간_EY")) else "",
          "delivery_date":    f"{r['양도_SY']:.0f}-{r['양도_SM']:02.0f}-{r['양도_SD']:02.0f}" if pd.notna(r.get("양도_SY")) else "",
          # --- 계약서에 인쇄되지 않음: 출처 정보로만 보관, 정답으로 쓰지 말 것 ---
          "meta_copyright_holder_name": r.get("저작권자명"),
          # --- 저작물측 정답 (공유마당/KOGL) ---
          "work_title_indexed": gm.get("title") or km.get("title") or "",
          "gt_file_size": best["size"] if best else "",
          "gt_digital_format": ext.lstrip(".").upper() if ext else "",
          "gt_file_swapped": bool(best and gm.get("gt_file") and gm["gt_file"]!=best["path"]),
          "gt_source_record": gm.get("gt_file","") and f"/mnt/d/copyright_dataset_metadata/{gm['media']}/{gm['bucket']}/ground_truth.jsonl",
        })

    df=pd.DataFrame(rows)

    # ---------- 내용 중복 (같은 크기 그룹만 해시 — 320GB 전수 해시는 불가) ----------
    grp=defaultdict(list)
    for i,rr in df[df.work_path!=""].iterrows(): grp[rr.work_bytes].append(i)
    def sig(p):
        h=hashlib.sha256()
        with open(p,"rb") as f: h.update(f.read(1<<18))
        return h.hexdigest()[:16]
    dup=Counter(); df["work_content_dup_group"]=""
    for size,ii in grp.items():
        if len(ii)<2: continue
        by=defaultdict(list)
        for i in ii:
            try: by[sig(df.at[i,"work_path"])].append(i)
            except OSError: pass
        for s,members in by.items():
            if len(members)>1:
                g=f"dup_{size}_{s[:8]}"
                for k,i in enumerate(members):
                    df.at[i,"work_content_dup_group"]=g
                    if k>0:
                        df.at[i,"exclude_reason"]=(df.at[i,"exclude_reason"]+";" if df.at[i,"exclude_reason"] else "")+"DUP_WORK_CONTENT"
                        df.at[i,"eval_ready"]=False
                        if df.at[i,"status"]=="OK": df.at[i,"status"]="OK_WITH_WARNING"
                dup[g]=len(members)

    df=df.sort_values(["media_declared","license_bucket","set_id"])
    df.to_csv(out,index=False,encoding="utf-8-sig")

    print(f"WROTE {out}   {len(df):,} rows × {len(df.columns)} cols")
    print("\n=== status ===");        print(df.status.value_counts().to_string())
    print(f"\neval_ready: {int(df.eval_ready.sum()):,} / {len(df):,}")
    print("\n=== exclude_reason ===");print(df[df.exclude_reason!=""].exclude_reason.value_counts().to_string())
    print("\n=== media_declared × license_bucket ===")
    print(pd.crosstab(df.media_declared,df.license_bucket,margins=True).to_string())
    print("\n=== work_source ===");   print(df.work_source.replace("","(none)").value_counts().to_string())
    print(f"\ncontent-dup groups: {len(dup)}  rows demoted: {sum(v-1 for v in dup.values())}")
    print(f"gt_file_swapped: {int(df.gt_file_swapped.sum()):,}   fits_zip_cap: {int(df.fits_zip_cap.sum()):,}")
    print(f"total work payload: {df.work_bytes.sum()/1e9:.1f} GB")


def main():
    ap = argparse.ArgumentParser(description="계약서-저작물 매칭 매니페스트 생성")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    idx, gt_meta, kogl_meta = build_work_index()
    build_manifest(idx, gt_meta, kogl_meta, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
