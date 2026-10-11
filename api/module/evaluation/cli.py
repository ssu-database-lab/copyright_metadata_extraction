"""평가 하네스 CLI — 매니페스트 기반 배치 채점.

사용(v3 — 계약서+저작물 쌍, plan/tta 채점):
  # 사전 점검(파일 존재·중복 ID) — API 호출 없음
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl --validate-only

  # 파일럿 20세트
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl --limit 20

  # 층화 400세트 + 비용 상한
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl \
      --workers 6 --max-cost 60000

  # TTA 표준 14속성 기준 채점 (정답셋도 TTA 이름이어야 한다)
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl --scoring-set tta

  # 집계만 다시(이미 돌린 results.jsonl 재사용)
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl --report-only

사용(v4 — 다리 하나, 23키 기록 + M1 평가기; docs/v4/M2_runner_spec.md §1.B B5):
  # T-001 계약서 단독 5세트 → test_predictions/ + (gold 가 있으면) test_metrics.json · report.md
  python -m api.module.evaluation.cli --manifest /tmp/v4_m2/t001_new5/manifest.jsonl --test T-001 \
      --out /tmp/v4_m2/t001_new5 --gold dataset/test_set_v4/gold23_t001_new5.jsonl [--workers 8] [--tau 0.65]

  # T-002 — 고정 목록 csv 에서 매니페스트를 바로 만들어 돌린다 (저작물 leg 만, 계약서는 규칙 문맥)
  python -m api.module.evaluation.cli --ids-csv dataset/test_set_v4/t002_ids.csv --work-ids 13385261,87248 \
      --test T-002 --out /tmp/v4_m2/t002_dev --gold dataset/test_set_v4/gold23_t002.jsonl [--consolidate-work]

  # 평가만 다시 (이미 기록한 test_predictions/ 재사용, 모델 호출 없음)
  python -m api.module.evaluation.cli --manifest ... --test T-001 --out ... --gold ... --report-only

`--test` 를 주면 채점 기준은 eval23 이다. 매니페스트 검증(`ok`)에 실패하면 **시작하지 않는다**(종료코드 2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# resolve() 는 /home/…/copyright 심볼릭 링크를 /mnt/c(OneDrive, drvfs) 로 풀어 버린다 — 그러면 module.* 로 올라온
# 모듈의 __file__ 과 매니페스트의 계약서 경로가 전부 /mnt/c 가 된다. 링크를 풀지 않는 abspath 로 둔다.
_HERE = Path(os.path.abspath(__file__))
_API_ROOT = _HERE.parents[2]          # .../api
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from module.evaluation.batch_runner import EVAL23, BatchRunner, RunConfig      # noqa: E402
from module.evaluation.manifest import (VALID_TESTS, load_manifest, load_ground_truth,  # noqa: E402
                                        validate_manifest)
from module.evaluation.report import aggregate, load_results, render_markdown  # noqa: E402


def build_orchestrator(out_dir: Path):
    """배포 파이프라인과 동일한 오케스트레이터. eval_e2e_contracts.py 와 같은 구성."""
    from api import ner_predict                      # noqa: 무거운 import (torch)
    from module.llm_extraction import LLMExtractionProcessor
    from web.pipeline import PipelineOrchestrator
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return PipelineOrchestrator(
        llm_processor=LLMExtractionProcessor(output_dir=str(runs / "llm_results")),
        ner_predict_fn=ner_predict,
        available_ner_models={"klue-roberta-large": {
            "name": "klue/roberta-large", "display_name": "KLUE RoBERTa Large"}},
        upload_dir=runs / "uploads", results_dir=runs / "results",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="계약서+저작물 세트 배치 평가 (v3 plan/tta · v4 eval23)")
    src = ap.add_argument_group("입력")
    src.add_argument("--manifest", default=None, help="manifest.jsonl (--ids-csv 와 둘 중 하나)")
    src.add_argument("--ids-csv", default=None,
                     help="v4 고정 목록 csv(dataset/test_set_v4/t00{1,2}_ids.csv) — <out>/manifest.jsonl 을 만들어 쓴다 (--test·--out 필요)")
    src.add_argument("--work-ids", default=None, help="--ids-csv 와 함께: 쉼표로 구분한 work_id 만 담는다")
    ap.add_argument("--ground-truth", default=None,
                    help="(v3) 기본값: 매니페스트와 같은 폴더의 ground_truth.jsonl")
    ap.add_argument("--out", default=None,
                    help="기본값: 매니페스트 폴더의 _eval_out/ (--ids-csv 에서는 필수)")
    ap.add_argument("--workers", type=int, default=None, help="동시 세트 수 (기본: v3 4 · v4 8)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-cost", type=float, default=None, help="누적 추정비용 상한(원)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--vlm-prefer", default="auto", choices=["auto", "gemma", "qwen"],
                    help="이미지 VLM 체인 (기본 auto: qwen3.8-omni-flash → qwen3.7-plus → Gemma 4(Google))")
    # 단계별 모델 — 기본값은 /api/llm-extract 단일처리와 동일하다.
    ap.add_argument("--model-name", default="alibaba-qwen3.5-122b-a10b",
                    help="LLM 메타데이터 추출 모델")
    ap.add_argument("--ocr-provider", default="alibaba",
                    choices=["alibaba", "mistral", "google", "naver"])
    ap.add_argument("--ocr-model", default=None,
                    help="미지정 시 provider 기본값 (alibaba → qwen3.7-plus, 생각 끔)")
    ap.add_argument("--ner-model", default="klue-roberta-large")
    ap.add_argument("--no-consolidate", action="store_true",
                    help="통합검증 생략 — 비용 최대 40%% 절감, 단 필드별 판정·신뢰도가 사라진다")
    ap.add_argument("--consolidation-model", default="alibaba-qwen3.5-122b-a10b")
    # 채점 기준 — plan(연구개발계획서 11속성) / tta(TTA 표준 14속성) / eval23(v4 23키 기록, M1 평가기).
    # 웹(/v3)에만 선택지가 있고 CLI 에는 없어서, TTA 정답셋을 CLI 로 돌리면
    # 11속성으로 대조되어 전 항목이 미채점으로 빠진다(정확도 '—'). 같은 코어를
    # 쓴다고 해 놓고 결과가 갈리는 지점이라 여기에도 둔다.
    ap.add_argument("--scoring-set", default="plan", choices=["plan", "tta", EVAL23],
                    help="채점 기준 (기본 plan = 계획서 11속성; --test 를 주면 eval23)")
    v4 = ap.add_argument_group("v4 (eval23)")
    v4.add_argument("--test", default=None, choices=list(VALID_TESTS),
                    help="시험 구분 — 주면 채점 기준이 eval23 이 된다 (엔트리의 test 가 우선)")
    v4.add_argument("--gold", default=None, help="gold23.jsonl — 주면 끝에 M1 평가기로 test_metrics.json·report.md 를 쓴다")
    v4.add_argument("--consolidate-work", action="store_true",
                    help="저작물 leg 에도 중재자(통합검증)를 돌린다 — 기본 생략 (저작물 leg 비용의 97%%)")
    v4.add_argument("--tau", type=float, default=0.65, help="설명 코사인 임계값 (평가기로 전달, 기본 0.65)")
    v4.add_argument("--frame-schedule", default="v4", choices=["v3", "v4"],
                    help="영상 키프레임 스케줄 (기본 v4: <900s 16장 · ≥900s 24장)")
    # 평가기(M1 §5·§7) 옵션 — τ 처럼 그대로 넘긴다. --report-only 재평가에서 임베딩 캐시를 다시 쓰려면 필요하다.
    v4.add_argument("--content-check", default="on", choices=["on", "off"],
                    help="설명 판정의 내용 토큰 검사 (평가기로 전달, 기본 on)")
    v4.add_argument("--embed-cache", default=None, help="임베딩 캐시 jsonl (평가기로 전달)")
    v4.add_argument("--embed-revision", default=None, help="KURE-v1 HF revision 고정 (평가기로 전달, 산출물에 기록)")
    v4.add_argument("--n-boot", type=int, default=None, help="부트스트랩 반복 수 (평가기로 전달, 기본 1000)")
    v4.add_argument("--title-fallback", action="store_true",
                    help="T-002: 계약서가 없는 세트에 매니페스트 제목을 문맥으로 준다 (본 시험 흉내 — 개발·파일럿용)")
    v4.add_argument("--only-predicted", action="store_true",
                    help="평가기 gold 를 예측이 있는 set_id 로 제한 (개발·파일럿 런 — 4,000세트 전부를 FN 으로 세지 않음). 기본 off")
    return ap


def _print_validation(v: dict, n_gt: Optional[int] = None) -> None:
    print(f"매니페스트 {v['total']}세트" + (f" · 정답 {n_gt}건" if n_gt is not None else ""))
    print(f"  구성 — 계약서단독 {v['contract_only']} · 저작물단독 {v['work_only']} · 쌍 {v['pairs']}"
          f" · 둘다없음 {v['neither']}")
    print(f"  중복 ID {v['duplicate_count']} · 저작물 파일 없음 {v['missing_work_count']} "
          f"· 계약서 파일 없음 {v['missing_contract_count']}")
    for p in v.get("problems") or []:
        print(f"  ⚠️ {p}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    eval23 = bool(args.test) or args.scoring_set == EVAL23
    scoring_set = EVAL23 if eval23 else args.scoring_set

    if not args.manifest and not args.ids_csv:
        print("⛔ --manifest 또는 --ids-csv 중 하나가 필요합니다", file=sys.stderr)
        return 2
    if args.ids_csv:
        if not args.test or not args.out:
            print("⛔ --ids-csv 에는 --test 와 --out 이 필요합니다", file=sys.stderr)
            return 2
        from module.evaluation.manifest_v4 import build_manifest_v4
        ids = [s.strip() for s in (args.work_ids or "").split(",") if s.strip()] or None
        mpath = build_manifest_v4(args.ids_csv, args.test, Path(args.out) / "manifest.jsonl", work_ids=ids)
        print(f"매니페스트 생성 → {mpath}")
    else:
        mpath = Path(args.manifest)

    entries = load_manifest(mpath)
    out_dir = Path(args.out) if args.out else mpath.parent / "_eval_out"
    v = validate_manifest(entries, test=args.test)

    gt = {}
    if not eval23:
        gt_path = Path(args.ground_truth) if args.ground_truth else mpath.parent / "ground_truth.jsonl"
        gt = load_ground_truth(gt_path)
    _print_validation(v, None if eval23 else len(gt))

    if not eval23:
        missing_gt = [e.set_id for e in entries if e.set_id not in gt]
        if missing_gt:
            print(f"  ⚠️ 정답 없는 세트 {len(missing_gt)}건 (예: {missing_gt[:5]})")

        # 채점 기준과 정답셋 속성 이름이 어긋나면 전 항목이 미채점으로 빠지고 결과가
        # 0%/'—' 로 보인다 — 오류가 아니라 "다 틀렸다"처럼 읽힌다. 돌리기 전에 잡는다.
        from module.evaluation.scoring import attributes_for
        want = {a.name for a in attributes_for(scoring_set)}
        have = set()
        for rec in list(gt.values())[:20]:
            # attributes 가 null 인 줄이 섞이면 set(None) 로 죽는다 — 정답셋 점검이
            # 실행을 막아서는 안 된다. 빈 dict 로 보고 넘어간다.
            have |= set((rec or {}).get("attributes") or {})
        overlap = len(want & have)
        print(f"  채점 기준 {scoring_set} ({len(want)}속성) · 정답셋 대조 {overlap}/{len(want)}")
        if have and overlap < len(want):
            miss = sorted(want - have)
            print(f"  ⚠️ 정답셋에 없는 채점 속성 {len(miss)}개: {miss[:6]} — "
                  f"미채점으로 빠져 정확도가 실제보다 높게 보일 수 있습니다"
                  + ("  (기준이 맞는지 --scoring-set 을 확인하세요)" if not overlap else ""))
    else:
        tests = sorted({(e.test or args.test or "(추론)") for e in entries})
        print(f"  채점 기준 eval23 · 시험 {tests} · gold {args.gold or '(없음 — 예측만 기록)'}")

    if args.validate_only:
        return 0 if v["ok"] else 1
    # 검증에 실패하면 시작하지 않는다(M2 B4) — 계약서 leg 요금을 쓴 뒤 open("") 에서 죽고
    # 재개마다 다시 과금되던 경로다. --report-only 는 모델을 부르지 않으므로 예외다.
    if not v["ok"] and not args.report_only:
        print("⛔ 매니페스트 검증 실패 — 실행하지 않습니다 (위 사유를 고친 뒤 다시 돌리세요)")
        return 2

    workers = args.workers or (8 if eval23 else 4)
    cfg = RunConfig(workers=workers, limit=args.limit,
                    resume=not args.no_resume, max_cost_krw=args.max_cost,
                    vlm_prefer=args.vlm_prefer, model_name=args.model_name,
                    ocr_provider=args.ocr_provider, ocr_model=args.ocr_model,
                    ner_model=args.ner_model,
                    consolidate=not args.no_consolidate,
                    consolidation_model=args.consolidation_model,
                    scoring_set=scoring_set,
                    test=args.test, consolidate_work=args.consolidate_work,
                    frame_schedule=args.frame_schedule, gold=args.gold, tau=args.tau,
                    content_check=(args.content_check == "on"), embed_cache=args.embed_cache,
                    embed_revision=args.embed_revision, n_boot=args.n_boot,
                    only_predicted=args.only_predicted, title_fallback=args.title_fallback)

    def cb(ev):
        if ev["type"] == "progress" and (ev["i"] % 10 == 0 or not ev["ok"] or eval23):
            acc = ev.get("accuracy")
            acc_s = f"{acc*100:.0f}%" if acc is not None else "—"
            print(f"  [{ev['i']}/{ev['total']}] {ev['set_id']} "
                  f"{'OK' if ev['ok'] else 'FAIL'} acc={acc_s} "
                  f"₩{ev['cost_krw_total']:,.0f} 누적실패={ev['failed']}"
                  + (f" — {ev['error']}" if ev.get("error") else ""), flush=True)
        elif ev["type"] == "cost_limit":
            print(f"  ⛔ 비용 상한 도달 ₩{ev['cost_krw']:,.0f} — 중단", flush=True)

    if not args.report_only:
        print(f"  모델 구성 — OCR {cfg.ocr_provider}/{cfg.ocr_model or '(기본)'} · "
              f"추출 {cfg.model_name} · NER {cfg.ner_model} · "
              f"통합검증 {cfg.consolidation_model if cfg.consolidate else '생략'} · "
              f"이미지VLM {cfg.vlm_prefer} · 채점 {cfg.scoring_set}"
              + (f" · 저작물 중재자 {'켬' if cfg.consolidate_work else '생략'} · 프레임 {cfg.frame_schedule}"
                 if eval23 else "")
              + f" · workers {cfg.workers}")

        runner = BatchRunner(build_orchestrator(out_dir), cfg, out_dir, progress_cb=cb)

        # Ctrl-C 로 실제로 멈춰야 한다. 기본 동작으로는 이미 제출된 세트가 전부 실행되어
        # 중단했다고 생각한 뒤에도 요금이 계속 발생한다.
        import signal

        def _sigint(signum, frame):
            print("\n  ⛔ 중단 요청 — 진행 중인 세트만 마치고 종료합니다 "
                  "(결과는 저장되며 --resume 으로 이어집니다)", flush=True)
            runner.cancel()
        signal.signal(signal.SIGINT, _sigint)

        summary = runner.run(entries, gt)
        print(f"\n실행 요약: {json.dumps(summary, ensure_ascii=False)}")
    else:
        runner = BatchRunner(None, cfg, out_dir, progress_cb=cb) if eval23 else None

    if eval23:
        fin = runner.finalize_eval23()
        print(f"\nv4 요약: 예측 {fin['predictions']}건 (정상 {fin['predictions_ok']}) · test={fin['test']}"
              + (f" · gold 제외 {fin.get('n_gold_dropped')}건 (--only-predicted)" if fin.get("only_predicted") else ""))
        if fin.get("evaluated"):
            mf, mi = fin.get("macro_f1"), fin.get("micro_f1")
            print(f"  Macro-F1 {mf*100:.1f}% · Micro-F1 {mi*100:.1f}% → {'합격' if fin.get('passed') else '불합격'}"
                  if mf is not None and mi is not None else "  지표 없음(채점 가능한 키 없음)")
            for k, f1 in (fin.get("per_key_f1") or {}).items():
                print(f"    {k:28s} {f1*100:6.1f}%" if f1 is not None else f"    {k:28s}      —")
            print(f"  출력: {out_dir}/test_metrics.json · report.md · per_set.jsonl")
        elif fin.get("error"):
            print(f"  ⚠️ 평가 실패: {fin['error']}")
        elif fin.get("warning"):
            print(f"  ⚠️ {fin['warning']}")
        print(f"\n산출물: {fin['pred_dir']} · {fin['pred_jsonl']} · {fin['log']} · {out_dir}/report.md")
        return 0

    results = load_results(out_dir / "results.jsonl")
    agg = aggregate(results)
    md = render_markdown(agg)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("\n" + md)
    print(f"\n산출물: {out_dir}/report.md · report.json · results.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
