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

# ---------------------------------------------------------------- 2026-10-31 변형 — 현행(=v2) 프롬프트에 덧대는 조각
# 현행 시스템 프롬프트(api.module.clip_extraction.vlm.prompts)는 2026-10-11 부터 위 v2 그대로다. 아래 변형은 그 문자열의
# 한 줄씩을 바꾼다. 예시 낱말은 시험 정답에서 가져오지 않는다(정답 지침·현행 예시의 낱말만).
IMG_DESC = '"description": "이미지 내용을 한국어로 2~3문장으로 객관적으로 설명",'
VID_DESC = ('"description": "영상 내용을 한국어 2~3문장으로 구체적으로 (주 장면과, 시작부터 끝까지 화면이 어떻게 바뀌는지 — '
            '카메라·피사체 이동, 장면 전환)",')
VID_BAN = ("★ 줄거리·사건 전개·등장인물의 역할·장르(범죄·액션·드라마·모험 등)는 쓰지 마세요. 설명·키워드·해시태그 모두\n"
           "   프레임에 실제로 보이는 대상·장소·행위·화면 특성(흑백, 자막, 구도 등)만으로 쓰세요.")
# 키워드 = 사이트 목록의 분류어 (Te-2 키워드 정답은 사이트 검색 태그: 대상·장소 이름, 상위 분류어, 영상의 연대·장르)
KW_CAT = ('"keywords": ["키워드 5~10개 — 아카이브 검색용 분류어. ① 저작물명이 알려 주는 대상·장소·행사 이름(예: 경복궁, 제주도) '
          '② 주 피사체와 그 상위 분류어를 함께(예: 고양이·동물, 소나무·나무·식물, 성당·건축물) ③ 장소·배경의 종류(공원, 바다, 도시, 산) '
          '④ 영상은 흑백이면 흑백영화, 화면·의상·자막으로 알 수 있으면 대략의 연대(예: 1930년대)와 장르(드라마·코미디·범죄·서부영화·다큐멘터리 등). '
          '부리·손잡이 같은 세부 부분, 색·조명·구도 낱말, 평가·추측 낱말은 쓰지 않는다"],')
# kwcat2 = kwcat + 형식 분류어(사진·그림·화풍)와 넓은 주제어(풍경·자연·인물·동물…) — 개발 세트에서 놓친 정답의 두 무리
KW_CAT2 = KW_CAT.replace('부리·손잡이 같은 세부 부분', '⑤ 작품 형식: 사진이면 사진, 그림이면 그림·회화와 화풍(수채화·유화·동양화·한국화·산수화·판화 등) '
                                                  '⑥ 넓은 주제어 1~2개(풍경·자연·인물·동물·식물·건축물·도시·역사 중 맞는 것). 부리·손잡이 같은 세부 부분')
# kwfilm = kwcat + 영상의 매체 낱말 '영상'과 화면·자막·저작물명으로 알 수 있는 이야기 소재(키워드에만)
KW_FILM = KW_CAT.replace("④ 영상은 흑백이면 흑백영화,", "④ 영상은 '영상'을 넣고, 흑백이면 흑백영화,").replace(
    "다큐멘터리 등). ", "다큐멘터리 등), 화면·자막·저작물명으로 알 수 있는 이야기 소재(범죄·결혼·전투·스포츠 등). ")
VID_BAN_CAT = ("★ 줄거리·사건 전개·등장인물의 역할은 쓰지 마세요. 설명·해시태그는 프레임에 실제로 보이는 대상·장소·행위·화면 특성"
               "(흑백, 자막, 구도 등)만으로 쓰고, 장르·연대는 키워드에만 쓸 수 있습니다.")
# 설명 = 정답 초안 지침 v0.3 (generate_descriptions.SYSTEM_PROMPT 규칙 1·2·5·6·6-1) — 같은 글쓰기 규칙
DESC_RULES = ("1~3문장, 공백 포함 30~200자, '-다'로 끝나는 현재형 평서문. 첫 문장 = 주 피사체 + 장소·배경 + 동작·상태 "
              "(저작물명이 알려 주는 대상·장소 이름은 화면과 모순되지 않으면 첫 문장에 쓴다). 이후 문장 = 구도·시점, 색·조명, 화면 속 글자")
IMG_DESC_G = ('"description": "' + DESC_RULES + ". '이 이미지는'으로 시작하지 않고, 평가·감상 어휘와 '~로 보인다' 같은 추측은 쓰지 않는다\",")
VID_DESC_G = ('"description": "' + DESC_RULES + ", 시간에 따른 변화(주 장면 1개와 시작→끝의 카메라·피사체 이동, 장면 전환). 줄거리는 쓰지 않는다. "
              "'이 영상은'으로 시작하지 않고, 평가·감상 어휘와 '~로 보인다' 같은 추측은 쓰지 않는다\",")
# descg2 = descg 를 1~2문장 · 60~120자로 (지침 범위 30~200 안 — 개발 세트 초안의 가운데 길이 79자 쪽으로)
IMG_DESC_G2 = IMG_DESC_G.replace("1~3문장, 공백 포함 30~200자", "1~2문장, 공백 포함 60~120자")
VID_DESC_G2 = VID_DESC_G.replace("1~3문장, 공백 포함 30~200자", "1~2문장, 공백 포함 60~120자")
# 해시태그 = 정답 초안 지침 v0.3 규칙 7 + 구도·배경·인물 표준 낱말 (현행 프롬프트 예시에 있는 낱말만)
HT_G = ("\"hashtags\": [\"해시태그 5~10개 — '#태그' 형식, 띄어쓰기 없이 붙여 쓴 명사형. 화면에 보이는 대상·장소·행위·매체 특성을 "
        "나타낸다. 저작물명의 대상·장소는 화면과 모순되지 않으면 넣는다. 구도·배경·인물은 짧은 표준 낱말로 쓴다(예: #정면구도 #초록배경 "
        "#상반신 #실내 #야외 #흑백사진 #드론 #항공). 개인 이름·나이, 라이선스·출처·기증 낱말, 평가어, 장르·분위기 태그는 쓰지 않는다. "
        "설명과 같은 근거(화면)에서만 뽑는다\"],")

VARIANTS: dict[str, dict[str, Any]] = {
    "prod": {},                                                              # 현행 그대로 (기준)
    "kwcat": {"kw": KW_CAT, "vid_ban": VID_BAN_CAT},                          # 키워드만 분류어 방식
    "descg": {"desc": (IMG_DESC_G, VID_DESC_G)},                              # 설명만 정답 지침 규칙
    "htg": {"ht": HT_G},                                                     # 해시태그만 정답 지침 규칙 + 표준 낱말
    "all": {"kw": KW_CAT, "vid_ban": VID_BAN_CAT, "desc": (IMG_DESC_G, VID_DESC_G), "ht": HT_G},
    "kwcat2": {"kw": KW_CAT2, "vid_ban": VID_BAN_CAT},                       # kwcat + 형식 분류어 · 넓은 주제어
    "descg2": {"desc": (IMG_DESC_G2, VID_DESC_G2)},
    "kwfilm": {"kw": KW_FILM, "vid_ban": VID_BAN_CAT.replace("장르·연대는 키워드에만", "장르·연대·이야기 소재는 키워드에만")},                          # 설명 지침 규칙 + 1~2문장 60~120자
}


def build_prompts(variant: str) -> tuple[str, str, str]:
    """현행 시스템 프롬프트(이미지·영상)에 변형 조각을 덧댄다 → (이미지 프롬프트, 영상 템플릿, 계약서 참고 줄)."""
    from api.module.clip_extraction.vlm import prompts as P
    v = VARIANTS[variant]
    img, vid = P.USER_PROMPT, P._VIDEO_USER_TEMPLATE
    esc = lambda t: t.replace("{", "{{").replace("}", "}}")              # noqa: E731 — 영상 템플릿은 format 을 거친다
    for name, base, desc in (("image", img, IMG_DESC), ("video", vid, VID_DESC)):
        assert KW1 in base and HT1 in base and desc in base, f"{name} 프롬프트가 바뀌었다 — 변형 조각을 다시 맞춰야 한다"
    assert VID_BAN in vid
    if v.get("kw"):
        img, vid = img.replace(KW1, v["kw"]), vid.replace(KW1, esc(v["kw"]))
    if v.get("ht"):
        img, vid = img.replace(HT1, v["ht"]), vid.replace(HT1, esc(v["ht"]))
    if v.get("desc"):
        img, vid = img.replace(IMG_DESC, v["desc"][0]), vid.replace(VID_DESC, esc(v["desc"][1]))
    if v.get("vid_ban"):
        vid = vid.replace(VID_BAN, esc(v["vid_ban"]))
    return img, vid, P._CONTEXT_LINE


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
    out = Path(a.out) / (f"{a.variant}__{a.image_backend}_{a.video_backend}" + (f"__r{a.rep}" if a.rep > 1 else ""))
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
        if not ctx.get("title") and r.get("title"):       # 계약서가 없는 개발 저작물 — 시험에서는 계약서의 저작물명이 같은 값이다
            ctx["title"] = r.title
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


# ---------------------------------------------------------------- 설명 교정 — 낱말이 틀린 글자(예: 오리 → 오이)만 고친다
PROOF_SYSTEM = ("너는 아카이브 설명문의 교정자다. 시각 모델이 쓴 설명문에서, 같은 모델이 쓴 키워드·해시태그와 저작물명에 비추어 "
                "글자가 빠지거나 바뀌어 뜻이 달라진 낱말만 고친다. 내용을 더하거나 빼거나 다시 쓰지 않는다. JSON 객체 하나만 출력한다.")


def proof_prompt(ctx: dict, rec: dict) -> str:
    return (f"저작물명: {ctx.get('title') or '(없음)'}\n키워드: {', '.join(rec.get('keywords') or [])}\n"
            f"해시태그: {' '.join(rec.get('hashtags') or [])}\n설명문: {rec.get('description') or ''}\n\n"
            "설명문 안의 낱말이 키워드·해시태그·저작물명의 낱말과 한두 글자만 다르고 그 때문에 다른 대상이 되었으면 "
            "(예: 해시태그 #오리 인데 설명문에 '오이') 그 낱말만 고친다. 그 밖의 글자는 한 글자도 바꾸지 않는다. 고칠 것이 없으면 그대로 둔다.\n"
            '출력: {"description": "<교정한 설명문>", "fixes": [["<원래 낱말>", "<고친 낱말>"]]}')


# 외국 문자 섞임 — 한국어 설명에 한자(前景·样式)·영어 낱말(various)이 끼어드는 경우만 고친다 (qwen3.8-omni-flash 에서 6~12 %)
_HAN = re.compile(r"[\u4e00-\u9fff]")
_LATIN_WORD = re.compile(r"(?<![A-Za-z])[a-z]{3,}(?![A-Za-z])")
_QUOTED = re.compile(r"['\"‘’“”「」『』][^'\"‘’“”「」『』]{0,80}['\"‘’“”「」『』]")


def has_leak(text: str) -> bool:
    bare = _QUOTED.sub("", text or "")                 # 따옴표 안은 화면 글자를 옮긴 것일 수 있다
    return bool(_HAN.search(bare) or _LATIN_WORD.search(bare))


LEAK_SYSTEM = ("너는 한국어 설명문의 교정자다. 설명문에 섞여 든 한자·영어 낱말만 같은 뜻의 한국어 낱말로 바꾼다. "
               "따옴표 안의 글자(화면에서 읽은 글자)와 그 밖의 모든 글자는 한 글자도 바꾸지 않는다. JSON 객체 하나만 출력한다.")


def leak_prompt(rec: dict) -> str:
    return (f"설명문: {rec.get('description') or ''}\n\n"
            "따옴표 밖에 있는 한자·영어 낱말(예: 前景 → 전경, 样式 → 양식, various → 다양한)만 한국어로 바꾼다. 띄어쓰기도 그대로 둔다.\n"
            '출력: {"description": "<고친 설명문>", "fixes": [["<원래 낱말>", "<고친 낱말>"]]}')


def cmd_proof(a) -> None:
    from openai import OpenAI
    from api.module.evaluation.v4.pilot_tau import _env
    src = Path(a.src) / "preds.jsonl"
    out = Path(str(a.src).rstrip("/") + ("+proof" if a.mode == "words" else "+leaks"))
    out.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=_env("DASHSCOPE_BASE_URL") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                    api_key=_env("DASHSCOPE_API_KEY"), timeout=120, max_retries=2)
    recs = [json.loads(l) for l in src.open(encoding="utf-8")]
    recs = [r for r in recs if r.get("ok")]
    lock = threading.Lock()
    pj = out / "preds.jsonl"
    pj.write_text("", encoding="utf-8")

    def one(r):
        rec = dict(r)
        if a.mode == "leaks" and not has_leak(r.get("description") or ""):
            r = {}                                          # 섞임이 없으면 부르지 않는다
        if r.get("description"):
            for _ in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=a.model, temperature=0, max_tokens=800, extra_body={"enable_thinking": False},
                        messages=[{"role": "system", "content": PROOF_SYSTEM if a.mode == "words" else LEAK_SYSTEM},
                                  {"role": "user", "content": proof_prompt(r.get("context") or {}, r) if a.mode == "words"
                                   else leak_prompt(r)}])
                    m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
                    d = json.loads(m.group(0)) if m else {}
                    new = (d.get("description") or "").strip()
                    if new and abs(len(new) - len(r["description"])) <= 6:     # 다시 쓰기는 버린다 — 낱말 교정만
                        rec["description_before_proof"] = r["description"]
                        rec["description"] = new
                        rec["proof_fixes"] = d.get("fixes") or []
                    break
                except Exception as e:  # noqa: BLE001
                    rec["proof_error"] = f"{type(e).__name__}: {e}"
        with lock, pj.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, recs))
    rows = [json.loads(l) for l in pj.open(encoding="utf-8")]
    changed = [r for r in rows if r.get("description_before_proof") and r["description_before_proof"] != r["description"]]
    log.info("교정 %d건 중 %d건 바뀜 → %s", len(rows), len(changed), pj)
    for r in changed[:20]:
        log.info("  %s %s", r["set_id"], r.get("proof_fixes"))


# ---------------------------------------------------------------- 정답 · 채점
def dev_gold(ids: pd.DataFrame, drafts_path: Path) -> dict[str, dict]:
    """개발 세트 정답: 키워드 = gold23.keywords_cell(인덱스), 해시태그 = KOGL 원본은 사이트 해시태그, 그 외는 초안 A (시험과 같은 정책)."""
    from api.module.evaluation.v4.gold23 import collection_list_counts, is_kogl_collection_list, keywords_cell
    from api.module.evaluation.v4.normalize import split_tags
    idx = pd.read_csv("dataset/gt_merged_index.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig").set_index("work_id")
    # 시험 정답과 같은 규칙 — KOGL 컬렉션 공통 목록(5건 이상 같음)은 키워드·해시태그 정답으로 쓰지 않는다 (2026-10-10 이후)
    counts = collection_list_counts(dict(enumerate(idx.reset_index().to_dict("records"))))
    dr = {}
    for l in drafts_path.open(encoding="utf-8"):
        d = json.loads(l)
        dr[d["work_id"]] = d
    gold = {}
    for sid in ids.work_id:
        irow = idx.loc[sid].to_dict()
        kw = keywords_cell(irow, counts)
        ht = None
        site_ht = split_tags(irow.get("work_해시태그", ""))
        if (irow.get("source", "").startswith("공공누리") and site_ht
                and not is_kogl_collection_list(irow, "work_해시태그", site_ht, counts)):
            ht = {"value": split_tags(irow["work_해시태그"]), "state": "value", "kind": "site"}
        elif dr.get(sid, {}).get("status") == "ok" and dr[sid].get("draft_hashtags"):
            ht = {"value": dr[sid]["draft_hashtags"], "state": "value", "kind": "generated"}
        desc = {"value": None, "state": "na"}
        if dr.get(sid, {}).get("status") == "ok" and dr[sid].get("draft_text"):
            desc = {"value": dr[sid]["draft_text"], "state": "value", "kind": "generated"}
        gold[sid] = {"keywords": kw, "hashtags": ht or {"value": None, "state": "na"}, "description": desc}
    return gold


def kw_dev_gold(path: Path) -> dict[str, dict]:
    """키워드 개발 세트(kw_dev.py) — 사이트 키워드만 정답이다 (해시태그·설명은 이 세트에서 재지 않는다)."""
    na = {"value": None, "state": "na"}
    g = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        g[r["work_id"]] = {"keywords": {"value": r["keywords"], "state": "value"}, "hashtags": na, "description": na}
    return g


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


def score_preds(preds: dict[str, dict], gold: dict[str, dict], cells: dict[str, str], cut: Optional[int] = None,
                judge=None) -> dict[str, Any]:
    from api.module.evaluation.v4.matchers import match_description, match_list
    out: dict[str, Any] = {}
    if judge is not None and cut is None:          # 설명 — 평가기와 같은 판정(KURE 코사인 ≥ τ + 내용 낱말 검사)
        res, by_cell = [], {}
        for sid, g in gold.items():
            ge = g.get("description")
            if not ge or ge.get("state") == "na" or sid not in preds:
                continue
            m = match_description(preds[sid].get("description"), ge, judge)
            res.append(m)
            by_cell.setdefault(cells.get(sid, "?"), []).append(m)
        out["description"] = {"all": pooled(res), "by_cell": {c: pooled(v) for c, v in sorted(by_cell.items())}}
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
    if a.gold == "dev":
        gold = dev_gold(ids, Path(a.drafts))
    elif a.gold == "kwdev":
        gold = kw_dev_gold(Path(a.ids).with_name("gold.jsonl"))
    else:
        gold = test_gold(Path(a.gold))
    judge = None
    if any((g.get("description") or {}).get("state") == "value" for g in gold.values()):
        from api.module.evaluation.v4.embed_judge import DEFAULT_MODEL_ID, EmbedJudge
        judge = EmbedJudge(model_id=DEFAULT_MODEL_ID, tau=a.tau, content_check=True,
                           cache_path=str(Path(a.runs) / "embed_cache.jsonl"))
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
        for cut in (None,) + ((5, 6, 7, 8) if a.cuts else ()):
            s = score_preds(preds, gold, cells, cut, judge)
            de = (s.get("description") or {}).get("all") or {}
            table.append({"run": d.name, "cut": cut or "all", "n_pred": len(preds),
                          "kw_f1": round(s["keywords"]["all"]["f1"], 3), "kw_n": s["keywords"]["all"]["n"],
                          "ht_f1": round(s["hashtags"]["all"]["f1"], 3), "ht_n": s["hashtags"]["all"]["n"],
                          "desc_f1": round(de["f1"], 3) if de else None, "desc_n": de.get("n") if de else None,
                          "ht_generated": round(s["hashtags"]["by_kind"].get("generated", {}).get("f1", float("nan")), 3),
                          "ht_site": round(s["hashtags"]["by_kind"].get("site", {}).get("f1", float("nan")), 3)})
            if cut is None:
                (d / "score.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    df = pd.DataFrame(table)
    print(df.to_string(index=False))
    if a.save:
        df.to_csv(a.save, index=False, encoding="utf-8-sig")


def cmd_compare(a) -> None:
    """같은 저작물끼리 비교 — 모든 실행이 값을 낸 저작물만, 항목별 F1 과 기준 실행 대비 저작물별 이김·짐."""
    from api.module.evaluation.v4.matchers import match_description, match_list
    ids = pd.read_csv(a.ids, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if a.gold == "dev":
        gold = dev_gold(ids, Path(a.drafts))
    elif a.gold == "kwdev":
        gold = kw_dev_gold(Path(a.ids).with_name("gold.jsonl"))
    else:
        gold = test_gold(Path(a.gold))
    ex = set(filter(None, a.exclude.split(",")))
    gold = {k: v for k, v in gold.items() if k in set(ids.work_id) and k not in ex}
    runs = {}
    for name in a.names.split(","):
        pj = Path(a.runs) / name / "preds.jsonl"
        runs[name] = {json.loads(l)["set_id"]: json.loads(l) for l in pj.open(encoding="utf-8")}
    judge = None
    rows = []
    base = a.names.split(",")[0]
    for key in ("keywords", "hashtags", "description"):
        sids = [s for s, g in gold.items() if (g.get(key) or {}).get("state") == "value"
                and all((r.get(s) or {}).get(key) for r in runs.values())]
        if not sids:
            continue
        if key == "description" and judge is None:
            from api.module.evaluation.v4.embed_judge import DEFAULT_MODEL_ID, EmbedJudge
            judge = EmbedJudge(model_id=DEFAULT_MODEL_ID, tau=a.tau, content_check=True,
                               cache_path=str(Path(a.runs) / "embed_cache.jsonl"))
        per = {}
        for name, r in runs.items():
            res = [match_description(r[s][key], gold[s][key], judge) if key == "description"
                   else match_list(r[s][key] if isinstance(r[s][key], list) else [r[s][key]], gold[s][key]) for s in sids]
            per[name] = res
        for name, res in per.items():
            st = pooled(res)
            win = sum(1 for x, b in zip(res, per[base]) if (x.tp - x.fp - x.fn) > (b.tp - b.fp - b.fn))
            lose = sum(1 for x, b in zip(res, per[base]) if (x.tp - x.fp - x.fn) < (b.tp - b.fp - b.fn))
            rows.append({"key": key, "run": name, "n": len(sids), "f1": round(st["f1"], 3),
                         "P": round(st["tp"] / max(1, st["tp"] + st["fp"]), 3), "R": round(st["tp"] / max(1, st["tp"] + st["fn"]), 3),
                         f"vs {base} 이김": win if name != base else "", "짐": lose if name != base else ""})
    print(pd.DataFrame(rows).to_string(index=False))


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
    r.add_argument("--rep", type=int, default=1, help="같은 변형을 다시 돌린 회차 (2 이상이면 폴더 이름 끝에 __rN) — 실행마다의 흔들림을 잰다")
    s = sub.add_parser("score")
    s.add_argument("--runs", required=True)
    s.add_argument("--ids", required=True)
    s.add_argument("--gold", default="dev", help="dev (tag_dev_v4 초안 정답) | kwdev (kw_dev.py 의 gold.jsonl) | gold23 경로")
    s.add_argument("--tau", type=float, default=0.65)
    s.add_argument("--cuts", action="store_true", help="태그 앞 5~8개만 쓴 점수도 낸다")
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
    pr = sub.add_parser("proof", help="설명 교정 — <src>+proof/ 에 쓴다")
    pr.add_argument("--src", required=True)
    pr.add_argument("--model", default="qwen3.5-122b-a10b")
    pr.add_argument("--workers", type=int, default=8)
    pr.add_argument("--mode", default="words", choices=["words", "leaks"],
                    help="words = 태그와 한두 글자 다른 낱말 교정 · leaks = 한자·영어가 섞인 설명만 그 낱말을 한국어로")
    c = sub.add_parser("compare", help="같은 저작물끼리 변형 비교 (첫 이름이 기준)")
    c.add_argument("--runs", required=True)
    c.add_argument("--names", required=True, help="실행 폴더 이름(쉼표) — 첫 번째가 기준")
    c.add_argument("--ids", required=True)
    c.add_argument("--gold", default="dev")
    c.add_argument("--drafts", default="dataset/gold_descriptions/tag_dev_a/drafts.jsonl")
    c.add_argument("--exclude", default=",".join(EXAMPLE_IDS))
    c.add_argument("--tau", type=float, default=0.65)
    a = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score, "retag": cmd_retag, "proof": cmd_proof, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
