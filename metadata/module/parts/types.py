"""공통 데이터 타입."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Decision:
    """추출기 단계의 공통 결과 스키마."""
    label: str
    value: str
    sent_id: Optional[int] = None
    tok_id: Optional[int] = None
    source: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """텍스트 내 연속 위치."""
    start: int
    end: int
    text: str
