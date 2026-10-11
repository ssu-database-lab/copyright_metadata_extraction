#!/usr/bin/env python
"""키워드·해시태그 개선 실험 (docs/v4/tags_investigation_20261010.md).

시스템과 같은 VLM 백엔드(DashScope Qwen3-VL · Qwen3.5-Omni)에 프롬프트 변형을 넣어 돌리고,
평가기와 같은 정합 규칙(matchers.match_list)으로 키워드·해시태그 F1 을 잰다.
개발 세트(시험 목록 밖)에서 고르고, 파일럿 200 에서는 한 번만 확인한다.

  python -m api.module.evaluation.v4.tag_lab run   --variant v1 --ids dataset/tag_dev_v4/dev_ids.csv --out dataset/tag_dev_v4/runs
  python -m api.module.evaluation.v4.tag_lab score --runs dataset/tag_dev_v4/runs --gold dev --ids dataset/tag_dev_v4/dev_ids.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import pandas as pd

log = logging.getLogger("v4.tag_lab")

# ---------------------------------------------------------------- 프롬프트 변형
KW0 = '"keywords": ["핵심 키워드 5~7개"],'
HT0 = "\"hashtags\": [\"해시태그 5~10개 — '#태그' 형식, 화면에 보이는 피사체·장소·행위·매체 특성을 나타내는 명사만, 띄어쓰기 없이, 인명·라이선스·출처 단어는 제외\"],"
CTX0 = "참고(계약서): {items} — 설명·키워드는 화면 내용 기준으로 작성"

KW1 = ('"keywords": ["키워드 5~7개 — 저작물명에 주제어(대상·장소·행사·단어)가 있으면 화면과 모순되지 않는 한 먼저 넣고, '
       '나머지는 화면의 주 피사체·장소·행위를 나타내는 일반 명사. 장르·분위기·평가·추측 단어는 쓰지 않는다"],')
HT1 = ("\"hashtags\": [\"해시태그 5~10개 — '#태그' 형식, 띄어쓰기 없이 붙여 쓴 명사. 화면에 보이는 대상·장소·행위·매체 특성을 나타낸다. "
       "저작물명의 주제어(대상·단어·장소)는 화면과 모순되지 않으면 넣는다. 장르·분위기·평가·추측 태그(예: #액션 #여행 #감성 #자연 #일상 #풍경), "
       "인명·나이, 라이선스·출처·기증 단어는 쓰지 않는다\"],")
CTX1 = ("참고(계약서): {items} — 설명은 화면에 보이는 것만 쓰되, 저작물명이 알려 주는 주제어(대상·단어·장소)는 화면과 모순되지 않으면 "
        "설명·키워드·해시태그에 넣는다. 수어 영상이면 저작물명의 단어를 수어로 표현하는 영상이다.")

# 작성 예시: 개발 세트(시험 목록 밖) 4건 — 채점에서는 빼야 한다 (EXAMPLE_IDS).
EXAMPLE_IDS = ("13071193", "73165", "13311323", "13370701")
EXAMPLES = """키워드·해시태그 작성 예시 (형식과 고르는 방식만 참고하고, 내용은 이 저작물에 맞게 쓴다):
- 사진, 저작물명 '춘천댐 공사 전경' → 키워드: 춘천댐, 댐, 공사, 전경, 흑백사진, 철골, 설비, 강원도 / 해시태그: #춘천댐 #댐 #공사 #전경 #흑백사진 #철골 #설비 #계곡
- 사진, 저작물명 '성남 봉국사 대광명전' → 키워드: 봉국사, 대광명전, 사찰, 전각, 단청, 목조건물 / 해시태그: #사찰 #전각 #단청 #처마 #현판 #목조건물 #불교건축
- 영상, 저작물명 '전문용어_수어_교통_주취 운전 (酒醉運轉)' → 키워드: 수어, 주취운전, 운전, 교통, 전문용어 / 해시태그: #수어 #주취운전 #운전 #손동작 #정면구도 #초록배경 #검은셔츠 #상반신
- 영상, 저작물명 '정선군_산촌_마을' → 키워드: 정선, 산촌, 마을, 전경, 드론, 항공, 풍경, 강원도 / 해시태그: #정선군 #산촌 #마을 #드론 #항공 #산 #들판 #전경
규칙: 저작물명의 대상·장소를 먼저, 그다음 화면의 주 피사체(구체어와 그 상위어를 함께, 예: #춘천댐 #댐), 매체 특성(흑백사진·드론·항공), 구도·배경을 쉬운 일반 명사로 쓴다."""

VARIANTS: dict[str, dict[str, str]] = {
    "v0": {"kw": KW0, "ht": HT0, "ctx": CTX0},   # 현행 시스템 프롬프트
    "v1": {"kw": KW1, "ht": HT1, "ctx": CTX1},   # 정답 지침(v0.3 6-1·7)에 맞춘 태그 규칙 + 제목 주제어 허용
    "v2": {"kw": KW1, "ht": HT1, "ctx": CTX1, "ex": EXAMPLES},   # v1 + 작성 예시 4건
}


def build_prompts(variant: str) -> tuple[str, str, str]:
    from api.module.clip_extraction.vlm import prompts as P
    v = VARIANTS[variant]
    img, vid = P.USER_PROMPT, P._VIDEO_USER_TEMPLATE
    for base_name, base in (("image", img), ("video", vid)):
        assert KW0 in base and HT0 in base, f"{base_name} 프롬프트의 키워드·해시태그 줄을 못 찾음"
    img = img.replace(KW0, v["kw"]).replace(HT0, v["ht"])
    vid = vid.replace(KW0, v["kw"]).replace(HT0, v["ht"].replace("{", "{{").replace("}", "}}"))
    if v.get("ex"):
        tail = "JSON 객체 하나만 출력하세요."
        assert img.rstrip().endswith(tail) and vid.rstrip().endswith(tail)
        img = img.rstrip()[: -len(tail)] + v["ex"] + "\n\n" + tail
        vid = vid.rstrip()[: -len(tail)] + v["ex"].replace("{", "{{").replace("}", "}}") + "\n\n" + tail
    return img, vid, v["ctx"]


def with_context(base: str, ctx_line: str, title: Optional[str], work_type: Optional[str]) -> str:
    items = []
    if title:
        items.append(f"저작물명 '{' '.join(str(title).split())}'")
    if work_type:
        items.append(f"종별 '{work_type}'")
    return base if not items else base.rstrip() + "\n\n" + ctx_line.format(items=", ".join(items))


# ---------------------------------------------------------------- 실행
def cmd_run(a) -> None:
    from api.module.clip_extraction import keyframes as K
    from api.module.clip_extraction.contract_context import parse as parse_ctx
    from api.module.clip_extraction.vlm import extractor as X
    from api.module.clip_extraction.vlm.prompts import SYSTEM_PROMPT, VIDEO_SYSTEM_PROMPT
    ids = pd.read_csv(a.ids, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if a.work_ids:
        ids = ids[ids.work_id.isin(a.work_ids.split(","))]
    if a.limit:
        ids = ids.head(a.limit)
    img_p, vid_t, ctx_line = build_prompts(a.variant)
    img_backend = {"qwen": X.make_qwen_backend, "omni": X.make_omni_backend, "openrouter": X.make_openrouter_backend}[a.image_backend]()
    vid_backend = {"qwen": X.make_qwen_backend, "omni": X.make_omni_backend}[a.video_backend]()
    out = Path(a.out) / f"{a.variant}__{a.image_backend}_{a.video_backend}"
    out.mkdir(parents=True, exist_ok=True)
    frames_root = Path(a.frames)
    done = set()
    pj = out / "preds.jsonl"
    if pj.exists():
        done = {json.loads(l)["set_id"] for l in pj.open(encoding="utf-8") if '"ok": true' in l}
    lock = threading.Lock()

    def one(r):
        sid = r.work_id
        if sid in done:
            return
        ctx = {}
        if r.get("contract_pdf"):
            c = parse_ctx(r.contract_pdf)
            ctx = {"title": c.get("title"), "work_type": c.get("work_type")}
        try:
            if r.media == "image":
                path = r.work_file
                if path.lower().endswith(".pptx"):
                    from api.module.clip_extraction.pptx_render import render_first_slide
                    png = frames_root / sid / "slide1.png"
                    if not png.exists():
                        png.parent.mkdir(parents=True, exist_ok=True)
                        render_first_slide(path, png)
                    path = str(png)
                res = img_backend.extract(Path(path), SYSTEM_PROMPT, with_context(img_p, ctx_line, ctx.get("title"), ctx.get("work_type")), max_tokens=1024, temperature=0.0)
            else:
                fdir = frames_root / sid
                kf_p = fdir / "keyframes.json"
                if kf_p.exists():
                    kf = json.loads(kf_p.read_text(encoding="utf-8"))
                else:
                    fdir.mkdir(parents=True, exist_ok=True)
                    kf = K.extract_keyframes(r.work_file, str(fdir), schedule="v4")
                    kf_p.write_text(json.dumps({"frames": kf["frames"], "duration": kf.get("duration")}, ensure_ascii=False), encoding="utf-8")
                times = [f["t"] for f in kf["frames"]]
                user = vid_t.format(n=len(times), times=", ".join(f"{float(t):.0f}초" for t in times))
                res = vid_backend.extract_frames([Path(f["path"]) for f in kf["frames"]], VIDEO_SYSTEM_PROMPT, with_context(user, ctx_line, ctx.get("title"), ctx.get("work_type")),
                                                 max_tokens=1024, temperature=0.0, label=sid)
            parsed = res.parsed or {}
            rec = {"set_id": sid, "ok": bool(res.ok), "error": res.error if not res.ok else None, "model": getattr(res, "model_label", None) or getattr(res, "model", None),
                   "context": ctx, "keywords": parsed.get("keywords"), "hashtags": parsed.get("hashtags"), "description": parsed.get("description")}
        except Exception as e:  # noqa: BLE001
            rec = {"set_id": sid, "ok": False, "error": f"{type(e).__name__}: {e}", "context": ctx}
        with lock, pj.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, [r for _, r in ids.iterrows()]))
    n_ok = sum(1 for l in pj.open(encoding="utf-8") if '"ok": true' in l)
    log.info("%s: %d/%d ok → %s", out.name, n_ok, len(ids), pj)


# ---------------------------------------------------------------- 2단계: 텍스트 모델로 태그 다시 쓰기
RETAG_SYSTEM = ("너는 공유저작물 아카이브의 키워드·해시태그 작성자다. 시각 모델이 쓴 설명과 계약서의 저작물명·종별을 보고 "
                "키워드와 해시태그를 다시 쓴다. 설명에 없는 대상·사실을 지어내지 않는다. JSON 객체 하나만 출력한다.")


KW_EXTRA = (" 키워드에는 매체 낱말(사진·흑백사진·영상·그림 중 해당하는 것) 하나를 넣고, 사람이 주 피사체이면 '인물'을 넣는다. "
            "저작물명에 지명이 있으면 그 지명이 속한 시·도 이름을 함께 넣는다(예: 정선 → 강원도).")


def retag_prompt(media: str, ctx: dict, rec: dict, extra: bool = False) -> str:
    return (f"매체: {MEDIA_KO.get(media, media)}\n저작물명: {ctx.get('title') or '(없음)'}\n종별: {ctx.get('work_type') or '(없음)'}\n"
            f"시각 모델 설명: {rec.get('description') or ''}\n시각 모델 키워드: {', '.join(rec.get('keywords') or [])}\n"
            f"시각 모델 해시태그: {' '.join(rec.get('hashtags') or [])}\n\n"
            "규칙: 키워드 5~8개, 해시태그 6~9개('#태그' 형식, 띄어쓰기 없음). 저작물명의 주제어(대상·단어·장소)를 화면과 모순되지 않으면 먼저 쓰고, "
            "설명에 나온 주 피사체·장소·행위·매체 특성·구도·배경을 쉬운 일반 명사로 쓴다. 구체어와 그 상위어를 함께 쓴다. "
            "장르·분위기·평가·추측 단어, 인명·나이, 라이선스·출처·기증 단어는 쓰지 않는다." + (KW_EXTRA if extra else "") + "\n\n" + EXAMPLES +
            '\n\n출력: {"keywords": ["..."], "hashtags": ["#..."]}')


MEDIA_KO = {"image": "이미지", "video": "영상"}


def cmd_retag(a) -> None:
    from openai import OpenAI
    from api.module.evaluation.v4.pilot_tau import _env, _parse_label  # noqa: F401  (_env 재사용)
    ids = pd.read_csv(a.ids, dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("work_id")
    src = Path(a.src) / "preds.jsonl"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=_env("DASHSCOPE_BASE_URL") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", api_key=_env("DASHSCOPE_API_KEY"), timeout=120, max_retries=2)
    recs = {}
    for l in src.open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("ok"):
            recs[r["set_id"]] = r
    lock = threading.Lock()
    pj = out / "preds.jsonl"

    def one(item):
        sid, r = item
        msg = retag_prompt(ids.loc[sid, "media"], r.get("context") or {}, r, extra=a.kw_extra)
        rec = {"set_id": sid, "ok": False, "context": r.get("context"), "description": r.get("description"), "src": str(src)}
        for _ in range(3):
            try:
                resp = client.chat.completions.create(model=a.model, temperature=0, max_tokens=1500, extra_body={"enable_thinking": False},
                                                      messages=[{"role": "system", "content": RETAG_SYSTEM}, {"role": "user", "content": msg}])
                txt = resp.choices[0].message.content or ""
                m = re.search(r"\{.*\}", txt, re.S)
                d = json.loads(m.group(0)) if m else {}
                if d.get("keywords") or d.get("hashtags"):
                    rec.update({"ok": True, "keywords": d.get("keywords"), "hashtags": d.get("hashtags"), "model": a.model})
                    break
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
        with lock, pj.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, recs.items()))
    log.info("retag %d건 → %s", len(recs), pj)


# ---------------------------------------------------------------- 정답 · 채점
def dev_gold(ids: pd.DataFrame, drafts_path: Path) -> dict[str, dict]:
    """개발 세트 정답: 키워드 = gold23.keywords_cell(인덱스), 해시태그 = KOGL 원본은 사이트 해시태그, 그 외는 초안 A (시험과 같은 정책)."""
    from api.module.evaluation.v4.gold23 import keywords_cell
    from api.module.evaluation.v4.normalize import split_tags
    idx = pd.read_csv("dataset/gt_merged_index.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("work_id")
    dr = {}
    for l in drafts_path.open(encoding="utf-8"):
        d = json.loads(l)
        dr[d["work_id"]] = d
    gold = {}
    for sid in ids.work_id:
        irow = idx.loc[sid].to_dict()
        kw = keywords_cell(irow)
        ht = None
        if irow.get("source", "").startswith("공공누리") and split_tags(irow.get("work_해시태그", "")):
            ht = {"value": split_tags(irow["work_해시태그"]), "state": "value", "kind": "site"}
        elif dr.get(sid, {}).get("status") == "ok" and dr[sid].get("draft_hashtags"):
            ht = {"value": dr[sid]["draft_hashtags"], "state": "value", "kind": "generated"}
        gold[sid] = {"keywords": kw, "hashtags": ht or {"value": None, "state": "na"}}
    return gold


def test_gold(path: Path) -> dict[str, dict]:
    g = {}
    for l in path.open(encoding="utf-8"):
        r = json.loads(l)
        g[r["set_id"]] = {"keywords": r["keys"]["keywords"], "hashtags": r["keys"]["hashtags"]}
    return g


def pooled(results: list) -> dict[str, Any]:
    tp = sum(r.tp for r in results); fp = sum(r.fp for r in results); fn = sum(r.fn for r in results)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(results), "tp": tp, "fp": fp, "fn": fn, "f1": (2 * p * rc / (p + rc)) if p + rc else 0.0}


def score_preds(preds: dict[str, dict], gold: dict[str, dict], cells: dict[str, str], cut: Optional[int] = None) -> dict[str, Any]:
    from api.module.evaluation.v4.matchers import match_list
    out: dict[str, Any] = {}
    for key in ("keywords", "hashtags"):
        res, by_cell, by_kind = [], {}, {}
        for sid, g in gold.items():
            ge = g[key]
            if not ge or ge.get("state") == "na" or sid not in preds:
                continue
            pv = preds[sid].get(key) or []
            if isinstance(pv, str):
                pv = [pv]
            if cut:
                pv = pv[:cut]
            m = match_list(pv, ge)
            res.append(m)
            by_cell.setdefault(cells.get(sid, "?"), []).append(m)
            by_kind.setdefault(ge.get("kind") or "site_tags", []).append(m)
        out[key] = {"all": pooled(res), "by_cell": {c: pooled(v) for c, v in sorted(by_cell.items())}, "by_kind": {k: pooled(v) for k, v in by_kind.items()}}
    return out


def cmd_score(a) -> None:
    ids = pd.read_csv(a.ids, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    cells = dict(zip(ids.work_id, ids.cell))
    gold = dev_gold(ids, Path(a.drafts)) if a.gold == "dev" else test_gold(Path(a.gold))
    gold = {k: v for k, v in gold.items() if k in cells and k not in set(filter(None, a.exclude.split(",")))}
    table = []
    for d in sorted(Path(a.runs).iterdir()):
        pj = d / "preds.jsonl"
        if not pj.exists():
            continue
        preds = {}
        for l in pj.open(encoding="utf-8"):
            r = json.loads(l)
            if r.get("ok"):
                preds[r["set_id"]] = r
        for cut in (None, 5, 6, 7, 8):
            s = score_preds(preds, gold, cells, cut)
            table.append({"run": d.name, "cut": cut or "all", "n_pred": len(preds),
                          "kw_f1": round(s["keywords"]["all"]["f1"], 3), "kw_n": s["keywords"]["all"]["n"],
                          "ht_f1": round(s["hashtags"]["all"]["f1"], 3), "ht_n": s["hashtags"]["all"]["n"],
                          "ht_generated": round(s["hashtags"]["by_kind"].get("generated", {}).get("f1", float("nan")), 3),
                          "ht_site": round(s["hashtags"]["by_kind"].get("site", {}).get("f1", float("nan")), 3)})
            if cut is None:
                (d / "score.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    df = pd.DataFrame(table)
    print(df.to_string(index=False))
    if a.save:
        df.to_csv(a.save, index=False, encoding="utf-8-sig")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    r.add_argument("--ids", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--frames", default="dataset/tag_dev_v4/frames")
    r.add_argument("--image-backend", default="qwen", choices=["qwen", "omni", "openrouter"])
    r.add_argument("--video-backend", default="omni", choices=["qwen", "omni"])
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--work-ids", default="")
    s = sub.add_parser("score")
    s.add_argument("--runs", required=True)
    s.add_argument("--ids", required=True)
    s.add_argument("--gold", default="dev", help="dev | gold23_t002.jsonl 경로")
    s.add_argument("--drafts", default="dataset/gold_descriptions/tag_dev_a/drafts.jsonl")
    s.add_argument("--save", default="")
    s.add_argument("--exclude", default=",".join(EXAMPLE_IDS), help="채점에서 뺄 work_id (기본: 작성 예시 4건)")
    t = sub.add_parser("retag")
    t.add_argument("--src", required=True)
    t.add_argument("--ids", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model", default="qwen3.5-122b-a10b")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--kw-extra", action="store_true", help="키워드 규칙 추가(매체 낱말·인물·시도)")
    a = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score, "retag": cmd_retag}[a.cmd](a)


if __name__ == "__main__":
    main()
