"""v4 평가 CLI (docs/v4/M1_evaluator_spec.md §7 CLI).

  python -m api.module.evaluation.v4.evaluate --test T-002 \\
      --gold dataset/test_set_v4/gold23_t002.jsonl --pred test_predictions/ --out results/v4_t002 \\
      [--tau 0.65] [--content-check on|off] [--cells] [--embed-cache cache.jsonl] [--embed-revision <sha>]

함수로 쓰려면: from api.module.evaluation.v4.evaluate import evaluate
로그: `api` 패키지가 import 시 root 로거를 WARNING 으로 잡아 두어 logging.basicConfig 는 무시된다 → 이 패키지
(api.module.evaluation.v4.*) 로거에만 핸들러를 달아 INFO/DEBUG 가 CLI 에서 보이게 한다 (gold23.configure_logging 과 같은 방식).
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from .embed_judge import DEFAULT_MODEL_ID, DEFAULT_TAU
from .evaluator import PASS_MARK, _ci_str, _pct, evaluate

__all__ = ["evaluate", "build_parser", "configure_logging", "main"]

LOGGER_NAME = __package__ or "api.module.evaluation.v4"  # evaluator/embed_judge/matchers 로거의 부모


class _CurrentStderrHandler(logging.StreamHandler):
    """항상 '지금의' sys.stderr 에 쓴다 — 테스트(capsys)처럼 stderr 가 바뀌어도 닫힌 스트림을 붙들지 않는다."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # StreamHandler.__init__/setStream 이 대입해도 무시한다
        return None


def configure_logging(verbose: bool = False) -> None:
    """v4 패키지 로거에 stderr 핸들러를 (한 번만) 달고 레벨을 맞춘다. root 로 전파하면 api 의 root 핸들러가 중복 출력하므로 끊는다."""
    pkg = logging.getLogger(LOGGER_NAME)
    pkg.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, _CurrentStderrHandler) for h in pkg.handlers):
        handler = _CurrentStderrHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        pkg.addHandler(handler)
    pkg.propagate = False


def _on_off(v: str) -> bool:
    s = v.strip().lower()
    if s in ("on", "true", "1", "yes"):
        return True
    if s in ("off", "false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"on 또는 off 를 주세요: {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m api.module.evaluation.v4.evaluate",
        description="v4 23키 평가기 — 예측(test_predictions)을 gold23.jsonl 과 대조해 Macro/Micro-F1 을 산출한다.",
    )
    p.add_argument("--test", required=True, choices=["T-001", "T-002"], help="시험 구분")
    p.add_argument("--gold", required=True, help="gold23.jsonl 경로")
    p.add_argument("--pred", required=True, help="예측: <set_id>.json 디렉터리 또는 .jsonl")
    p.add_argument("--out", required=True, help="출력 디렉터리 (test_metrics.json, report.md, per_set.jsonl)")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU, help="설명 코사인 임계값 (기본 0.65)")
    p.add_argument("--content-check", type=_on_off, default=True, metavar="on|off", help="설명 내용 토큰 확인 (기본 on)")
    p.add_argument("--cells", action="store_true", help="report.md 에 셀(매체×라이선스)별 항목 표를 포함")
    p.add_argument("--embed-cache", default=None, help="임베딩 캐시 jsonl 경로")
    p.add_argument("--embed-revision", default=None, help="HF 모델 revision 고정 (결과에 기록)")
    p.add_argument("--embed-model", default=DEFAULT_MODEL_ID, help=f"임베딩 모델 id (기본 {DEFAULT_MODEL_ID})")
    p.add_argument("--device", default=None, help="cpu | cuda (기본 자동)")
    p.add_argument("--n-boot", type=int, default=1000, help="부트스트랩 재표본 수 (기본 1000)")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG 로그")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    m = evaluate(
        args.test, args.gold, args.pred, args.out,
        tau=args.tau, content_check=args.content_check, cells=args.cells,
        embed_cache=args.embed_cache, embed_revision=args.embed_revision,
        embed_model=args.embed_model, device=args.device, n_boot=args.n_boot,
    )
    print(f"[{args.test}] 세트 {m['n_sets']} · 예측 누락 {len(m['warnings']['missing_predictions'])}")
    print(f"  Macro-F1 {_pct(m['macro_f1'])}  (CI {_ci_str(m['macro'].get('ci95_f1'))})  Micro-F1 {_pct(m['micro_f1'])}")
    print(f"  합격 기준 {100 * PASS_MARK:.0f}% → {'합격' if m['passed'] else '불합격'}")
    print(f"  출력: {m['outputs']['metrics']} · {m['outputs']['report']} · {m['outputs']['per_set']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
