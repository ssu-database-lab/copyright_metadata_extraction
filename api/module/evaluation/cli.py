"""평가 하네스 CLI — 매니페스트 기반 배치 채점.

사용:
  # 사전 점검(파일 존재·중복 ID) — API 호출 없음
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl --validate-only

  # 파일럿 20세트
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl --limit 20

  # 층화 400세트 + 비용 상한
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl \
      --workers 6 --max-cost 60000

  # 집계만 다시(이미 돌린 results.jsonl 재사용)
  python -m module.evaluation.cli --manifest eval_sets/manifest.jsonl --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_API_ROOT = _HERE.parents[2]          # .../api
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from module.evaluation.batch_runner import BatchRunner, RunConfig      # noqa: E402
from module.evaluation.manifest import (load_manifest, load_ground_truth,  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser(description="계약서+저작물 세트 배치 평가")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ground-truth", default=None,
                    help="기본값: 매니페스트와 같은 폴더의 ground_truth.jsonl")
    ap.add_argument("--out", default=None,
                    help="기본값: 매니페스트 폴더의 _eval_out/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-cost", type=float, default=None, help="누적 추정비용 상한(원)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--vlm-prefer", default="gemma", choices=["gemma", "qwen"],
                    help="이미지 VLM 우선순위 (기본 gemma → qwen 폴백)")
    # 단계별 모델 — 기본값은 /api/llm-extract 단일처리와 동일하다.
    ap.add_argument("--model-name", default="alibaba-qwen3.5-122b-a10b",
                    help="LLM 메타데이터 추출 모델")
    ap.add_argument("--ocr-provider", default="alibaba",
                    choices=["alibaba", "mistral", "google", "naver"])
    ap.add_argument("--ocr-model", default=None,
                    help="미지정 시 provider 기본값 (alibaba → qwen3-vl-235b-a22b-instruct)")
    ap.add_argument("--ner-model", default="klue-roberta-large")
    ap.add_argument("--no-consolidate", action="store_true",
                    help="통합검증 생략 — 비용 최대 40%% 절감, 단 필드별 판정·신뢰도가 사라진다")
    ap.add_argument("--consolidation-model", default="alibaba-qwen3.5-122b-a10b")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    entries = load_manifest(mpath)
    gt_path = Path(args.ground_truth) if args.ground_truth else mpath.parent / "ground_truth.jsonl"
    gt = load_ground_truth(gt_path)
    out_dir = Path(args.out) if args.out else mpath.parent / "_eval_out"

    v = validate_manifest(entries)
    print(f"매니페스트 {v['total']}세트 · 정답 {len(gt)}건")
    print(f"  중복 ID {v['duplicate_count']} · 저작물 없음 {v['missing_work_count']} "
          f"· 계약서 없음 {v['missing_contract_count']} · 계약서 미지정 {v['work_only']}")
    missing_gt = [e.set_id for e in entries if e.set_id not in gt]
    if missing_gt:
        print(f"  ⚠️ 정답 없는 세트 {len(missing_gt)}건 (예: {missing_gt[:5]})")
    if not v["ok"]:
        print(f"  ⚠️ 누락 예시 — 저작물 {v['missing_work'][:3]} / 계약서 {v['missing_contract'][:3]}")
    if args.validate_only:
        return 0 if v["ok"] else 1

    if not args.report_only:
        cfg = RunConfig(workers=args.workers, limit=args.limit,
                        resume=not args.no_resume, max_cost_krw=args.max_cost,
                        vlm_prefer=args.vlm_prefer, model_name=args.model_name,
                        ocr_provider=args.ocr_provider, ocr_model=args.ocr_model,
                        ner_model=args.ner_model,
                        consolidate=not args.no_consolidate,
                        consolidation_model=args.consolidation_model)
        print(f"  모델 구성 — OCR {cfg.ocr_provider}/{cfg.ocr_model or '(기본)'} · "
              f"추출 {cfg.model_name} · NER {cfg.ner_model} · "
              f"통합검증 {cfg.consolidation_model if cfg.consolidate else '생략'} · "
              f"이미지VLM {cfg.vlm_prefer}")

        def cb(ev):
            if ev["type"] == "progress" and (ev["i"] % 10 == 0 or not ev["ok"]):
                acc = ev.get("accuracy")
                acc_s = f"{acc*100:.0f}%" if acc is not None else "—"
                print(f"  [{ev['i']}/{ev['total']}] {ev['set_id']} "
                      f"{'OK' if ev['ok'] else 'FAIL'} acc={acc_s} "
                      f"₩{ev['cost_krw_total']:,.0f} 누적실패={ev['failed']}", flush=True)
            elif ev["type"] == "cost_limit":
                print(f"  ⛔ 비용 상한 도달 ₩{ev['cost_krw']:,.0f} — 중단", flush=True)

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
