from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Decision:
    """추출기 단계에서 공통으로 사용하는 결과 스키마."""

    label: str
    value: str
    sent_id: Optional[int] = None
    tok_id: Optional[int] = None
    source: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

