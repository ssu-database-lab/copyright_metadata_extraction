"""영상 모델 비교 채점기 — 공개 루브릭 기반 재현 가능 채점.

`_video_model_test/results.json` (8모델 × 5영상) 을 읽어 GOLD 설명에서 도출한
항목 리스트와 대조한다. 채점은 `description + keywords` 를 공백 제거 후
부분문자열로 매칭하며, 표기 흔들림은 alts 로 흡수한다.

사용:  python score_video_models.py [results.json 경로]
"""
import json, sys, math
from collections import defaultdict

# ── 루브릭 ────────────────────────────────────────────────────────────────
# 각 항목은 GOLD 설명(또는 튤립 영상처럼 GOLD가 행정정보인 경우 제목)에서 도출.
# alts = 모델이 쓸 수 있는 표기 변형. 오전사(파파이/퍼시발 등)는 의도적으로 제외 —
# 메타데이터로 저장될 값이 틀리면 적중으로 볼 수 없기 때문.
RUBRIC = {
    "시장의 반찬가게_002": {
        "_basis": "gold",
        "마늘":       ["마늘"],
        "깻잎":       ["깻잎"],
        "검정콩조림": ["검은콩", "검정콩", "콩조림", "콩자반", "블랙빈"],
        "반찬":       ["반찬", "밑반찬"],
        "플라스틱용기": ["플라스틱"],
        "시장":       ["시장", "마트", "반찬가게"],
    },
    "바람에 흔들리는 튤립 배경 소스": {
        # ⚠ GOLD가 기증 행정정보(기증식별번호·신청일시)라 시각 설명이 아님 → 제목으로 채점
        "_basis": "title",
        "튤립":       ["튤립"],
        "바람·흔들림": ["바람", "흔들"],
        "꽃밭/정원":  ["꽃밭", "정원", "화단", "화원"],
    },
    "Fright to the Finish 1954": {
        "_basis": "gold",
        "뽀빠이":     ["뽀빠이", "popeye"],          # 파파이·퍼시발·페퍼 = 오전사, 불인정
        "블루토":     ["블루토", "bluto"],
        "올리브":     ["올리브", "olive"],
        "유령/공포":  ["유령", "귀신", "공포", "할로윈", "halloween", "해골"],
        "장난":       ["장난", "짓궂", "골탕", "소동", "괴롭", "슬랩스틱"],
        "애니메이션": ["애니메이션", "만화", "cartoon", "animation"],
    },
    "어느 마을 어느 도로": {
        "_basis": "gold",
        "마을":   ["마을", "village", "촌락"],
        "산":     ["산", "언덕", "산맥"],
        "바다":   ["바다", "해안", "해변"],
        "도로":   ["도로", "길", "road"],
        "구름":   ["구름", "흐린", "흐림"],
        "차":     ["차량", "자동차", "승용차"],
    },
    "도시의 밤과 차": {
        "_basis": "gold",
        "지하철역": ["지하철", "역사", "전철", "승강장", "플랫폼"],
        "도로":     ["도로", "거리", "차도"],
        "오토바이": ["오토바이", "바이크", "스쿠터", "이륜"],
        "차":       ["차량", "자동차", "승용차"],
        "밤":       ["밤", "야간", "야경", "night"],
    },
}

# 단가 (USD / 1M tokens) — 출처가 확인된 모델만 기재. 미확인은 None → 비용 '—' 출력.
# ⚠ 단가는 실측이 아니라 공개 단가 인용이다. 토큰 수만 본 시험의 실측값.
PRICING = {
    "qwen3-vl-235b-a22b-instruct":  (0.40, 1.60, "Alibaba 공식(비용산정서 2026-07-15)"),
    "google/gemma-4-31b-it":        (0.10, 0.34, "OpenRouter(비용산정서)"),
    "qwen3.7-plus":                 (0.40, 1.60, "Alibaba 발표가(VentureBeat)"),
    "qwen3.5-omni-plus":            (0.40, 4.80, "3자 집계 — 계약 전 공식 확인 필요"),
    "qwen3.6-plus":                 None,
    "qwen3-vl-flash":               None,
    "qwen3.6-35b-a3b":              None,
    "qwen3-omni-flash":             None,
}
KRW = 1400  # 비용산정서와 동일 환율


def score_text(desc, kw, items):
    txt = (desc + " " + " ".join(kw)).replace(" ", "").lower()
    return [k for k, alts in items.items()
            if not k.startswith("_") and any(a.replace(" ", "").lower() in txt for a in alts)]


def main(path):
    R = json.load(open(path, encoding="utf-8"))
    per_model = defaultdict(lambda: {"hit": 0, "tot": 0, "sec": [], "in": 0, "out": 0, "kw": []})
    total_items = 0
    print("=" * 96)
    for v in R:
        title = v["video"]["title"]
        items = RUBRIC.get(title)
        if not items:
            print(f"⚠ 루브릭 없음: {title}")
            continue
        n_items = len([k for k in items if not k.startswith("_")])
        total_items += n_items
        p = v["probe"]
        cov = max(v["frame_times"]) / p["duration"] * 100 if p["duration"] else 0
        print(f"\n▶ {title}  ({p['duration']}s · {len(v['frames'])}프레임 · "
              f"마지막 프레임 {max(v['frame_times']):.1f}s = 앞 {cov:.0f}% 구간)")
        print(f"  채점기준({items['_basis']}) {n_items}항목: "
              f"{', '.join(k for k in items if not k.startswith('_'))}")
        rows = []
        for m in v["models"]:
            if m.get("error"):
                continue
            hit = score_text(m.get("desc", ""), m.get("kw", []), items)
            rows.append((len(hit), m, hit))
            s = per_model[m["model"]]
            s["hit"] += len(hit); s["tot"] += n_items; s["sec"].append(m["sec"])
            s["in"] += m["in"]; s["out"] += m["out"]; s["kw"].append(len(m.get("kw", [])))
        for n, m, hit in sorted(rows, key=lambda x: -x[0]):
            print(f"    {m['model'][:30]:<32}{n}/{n_items}  {m['sec']:>5.1f}s  {', '.join(hit)}")

    print("\n" + "=" * 96)
    print(f"■ 종합 (총 {total_items}항목)\n")
    print(f"  {'모델':<32}{'적중':>9}{'정확도':>8}{'평균초':>8}{'입력tok':>9}{'출력tok':>8}{'영상1건 ₩':>12}")
    print("  " + "-" * 94)
    final = []
    for mdl, s in per_model.items():
        acc = s["hit"] / s["tot"] * 100
        pr = PRICING.get(mdl)
        krw = ((s["in"] / 1e6 * pr[0] + s["out"] / 1e6 * pr[1]) / len(R) * KRW) if pr else None
        final.append((acc, mdl, s, krw))
    for acc, mdl, s, krw in sorted(final, key=lambda x: (-x[0], sum(x[2]["sec"]))):
        cost = f"₩{krw:.2f}" if krw is not None else "— (단가미확인)"
        print(f"  {mdl:<32}{s['hit']:>4}/{s['tot']:<4}{acc:>7.1f}%"
              f"{sum(s['sec'])/len(s['sec']):>8.1f}{s['in']:>9,}{s['out']:>8,}{cost:>14}")
    print("\n  ※ 토큰은 실측, 단가는 공개가 인용. 출력토큰이 모델별 8배 차이(629~5,271) —")
    print("     입력(프레임)은 거의 동일하므로 비용차는 사실상 출력토큰이 결정한다.")
    return final


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/mnt/d/copyright_dataset_metadata/_video_model_test/results.json")
