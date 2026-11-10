#!/usr/bin/env python
"""시험 세트 v5 → 단계별 내부 시험 묶음 (50 · 100 · 200 · 300 · 500 · 1000 · 1500 · 2000 · 2500 · 3000 · 4000).

시험 당일과 같은 PC·같은 화면(/test)에서 작은 묶음부터 차례로 돌려 오류를 잡기 위한 준비물을 만든다.

구성 (묶음 N):
  - 포개짐: 50 ⊂ 100 ⊂ … ⊂ 4000. 큰 묶음은 작은 묶음을 그대로 품는다.
  - 층화: 유형(만료·기증·CCL·KOGL) × 매체 × 템플릿(A/B/C) 비율이 어느 묶음에서나 전체와 같도록 고르게 섞은 순서의 앞 N 개.
  - Te-1 N = 공유 저작물(이미지·영상) 0.95N + 어문 0.05N
    Te-2 N = 같은 공유 저작물 0.95N + Te-2 전용 저작물 0.05N
    → Te-2 의 0.95N 은 같은 묶음의 Te-1 결과(1~16번)를 다시 쓰고, 0.05N 은 계약서를 직접 처리한다 (4000 에서 3,800 / 200).

출력 (--out, 기본 /mnt/e/시험자료_v5 = E:\\시험자료_v5):
  계약서/          4,200 계약서 PDF (복사)
  저작물/          4,000 저작물 (같은 E: 드라이브의 원본에 하드 링크 — 디스크를 더 쓰지 않는다. 이름은 원본 그대로 = Te-2 정답 file_name)
  answers/GNNNN.jsonl   묶음별 정답 (Te-1 + Te-2 레코드, gold_v5.jsonl 에서 그대로)
  lists/GNNNN.csv       묶음별 세트 목록 (점검용)
  README.md

화면에서는 세 경로를 같게 두고 정답 파일만 바꾼다. 정답이 있으면 정답에 있는 세트만 실행된다.

  python -m api.module.dataset_builder.build_progressive_groups_v5 [--out /mnt/e/시험자료_v5] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
V5 = ROOT / "dataset/test_set_v5"
SIZES = (50, 100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000, 4000)
SEED = "20261109"
BK_KO = {"expired": "만료", "donated": "기증", "ccl": "CCL", "kogl": "KOGL"}
MED_KO = {"text": "어문", "image": "이미지", "video": "영상"}


def _h(*parts: Any) -> str:
    return hashlib.sha256(":".join([SEED, *map(str, parts)]).encode()).hexdigest()


def interleave(items: list[dict], key) -> list[dict]:
    """층(key)마다 섞은 뒤, 층 안 순위의 상대 위치((순위+0.5)/층 크기)로 전체를 정렬한다 → 어떤 앞부분도 층 비율을 따른다."""
    strata: dict[Any, list[dict]] = defaultdict(list)
    for it in items:
        strata[key(it)].append(it)
    scored = []
    for k, group in strata.items():
        group.sort(key=lambda v: _h("in", v["work_id"]))
        for rank, v in enumerate(group):
            scored.append(((rank + 0.5) / len(group), _h("tie", v["work_id"]), v))
    return [v for _, _, v in sorted(scored, key=lambda x: (x[0], x[1]))]


def n_minor(n: int) -> int:
    """묶음 N 중 어문(Te-1) · Te-2 전용(Te-2) 몫 = 5 % (4000 → 200), 반올림."""
    return int(n / 20 + 0.5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/mnt/e/시험자료_v5")
    ap.add_argument("--dry-run", action="store_true", help="파일은 만들지 않고 구성만 출력")
    a = ap.parse_args(argv)
    out = Path(a.out)

    values = [json.loads(l) for l in open(V5 / "values.jsonl", encoding="utf-8")]
    gold = [json.loads(l) for l in open(V5 / "answers/gold_v5.jsonl", encoding="utf-8")]
    g1 = {g["set_id"]: g for g in gold if g["test"] == "T-001"}
    g2 = {g["set_id"]: g for g in gold if g["test"] == "T-002"}
    stratum = lambda v: (v["bucket"], v["media"], v["template"])  # noqa: E731
    shared = interleave([v for v in values if v["in_te1"] and v["in_te2"]], stratum)
    text = interleave([v for v in values if v["in_te1"] and not v["in_te2"]], stratum)
    only2 = interleave([v for v in values if v["in_te2"] and not v["in_te1"]], stratum)
    assert (len(shared), len(text), len(only2)) == (3800, 200, 200), (len(shared), len(text), len(only2))

    groups = {}
    for n in SIZES:
        k = n_minor(n)
        te1 = shared[: n - k] + text[:k]
        te2 = shared[: n - k] + only2[:k]
        assert len(te1) == len(te2) == n and all(v["work_id"] in g1 for v in te1) and all(v["work_id"] in g2 for v in te2)
        groups[n] = (te1, te2)

    # 구성 요약
    summary = []
    for n, (te1, te2) in groups.items():
        c1 = Counter((v["bucket"], v["media"]) for v in te1)
        c2 = Counter((v["bucket"], v["media"]) for v in te2)
        summary.append({
            "group": f"G{n:04d}", "n": n,
            "te1": {f"{BK_KO[b]}/{MED_KO[m]}": c for (b, m), c in sorted(c1.items())},
            "te2": {f"{BK_KO[b]}/{MED_KO[m]}": c for (b, m), c in sorted(c2.items())},
            "te1_templates": dict(sorted(Counter(v["template"] for v in te1).items())),
            "te2_templates": dict(sorted(Counter(v["template"] for v in te2).items())),
            "te2_reuses_te1": n - n_minor(n), "te2_contract_leg": n_minor(n),
            "te2_video_gb": round(sum(os.path.getsize(v["work_file"]) for v in te2 if v["media"] == "video") / 1e9, 1),
        })
    print(json.dumps([{k: s[k] for k in ("group", "te1_templates", "te2_reuses_te1", "te2_contract_leg", "te2_video_gb")}
                      for s in summary], ensure_ascii=False))
    if a.dry_run:
        return 0

    # 파일
    (out / "계약서").mkdir(parents=True, exist_ok=True)
    (out / "저작물").mkdir(exist_ok=True)
    (out / "answers").mkdir(exist_ok=True)
    (out / "lists").mkdir(exist_ok=True)
    copied = linked = 0
    for v in values:
        dst = out / "계약서" / v["pdf_name"]
        src = V5 / "계약서" / v["pdf_name"]
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            copied += 1
        if v["in_te2"]:
            wsrc = Path(v["work_file"])
            wdst = out / "저작물" / wsrc.name
            if not wdst.exists():
                try:
                    os.link(wsrc, wdst)                  # 같은 드라이브: 하드 링크 (추가 용량 0)
                except OSError:
                    shutil.copy2(wsrc, wdst)             # 다른 드라이브면 복사
                linked += 1
    by_id = {v["work_id"]: v for v in values}
    for n, (te1, te2) in groups.items():
        name = f"G{n:04d}"
        with open(out / "answers" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for v in te1:
                f.write(json.dumps(g1[v["work_id"]], ensure_ascii=False) + "\n")
            for v in te2:
                f.write(json.dumps(g2[v["work_id"]], ensure_ascii=False) + "\n")
        with open(out / "lists" / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["test", "set_id", "유형", "매체", "템플릿", "Te-1·Te-2 공유", "제목", "계약서 파일", "저작물 파일"])
            for test, rows in (("Te-1", te1), ("Te-2", te2)):
                for v in rows:
                    w.writerow([test, v["work_id"], BK_KO[v["bucket"]], MED_KO[v["media"]], v["template"],
                                "O" if (v["in_te1"] and v["in_te2"]) else "", v["제목"], v["pdf_name"],
                                Path(v["work_file"]).name if (test == "Te-2") else ""])
    (out / "groups.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"계약서 {copied} 복사 · 저작물 {linked} 연결 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
