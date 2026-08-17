"""평가(evaluation) 모듈 — 계약서+저작물 세트 배치 채점.

연구개발계획서 2-4 '메타데이터 속성 추출' 11속성에 대해 파이프라인 출력을
정답셋과 대조한다. CLI(harness)와 웹 배치 API가 **같은 코어**를 쓰므로
어느 경로로 돌리든 결과가 동일하다.

  manifest.py     세트 목록(계약서+저작물 쌍) 생성·적재·검증
  scoring.py      속성별 비교기 — 속성마다 채점 방식이 다르다
  batch_runner.py 동시 실행 + 체크포인트 + 재개 + 비용 상한
  report.py       집계 → 마크다운/JSON 리포트
"""

from .manifest import ManifestEntry, load_manifest, validate_manifest
from .scoring import ATTRIBUTES, score_set
from .batch_runner import BatchRunner, RunConfig
from .report import aggregate, render_markdown

__all__ = [
    "ManifestEntry", "load_manifest", "validate_manifest",
    "ATTRIBUTES", "score_set",
    "BatchRunner", "RunConfig",
    "aggregate", "render_markdown",
]
