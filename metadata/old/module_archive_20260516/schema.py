"""스키마 관련 유틸 — configs/labels.yaml 로딩."""
import yaml
from pathlib import Path
from typing import List


def load_schema_labels(config_path: str = "configs/labels.yaml") -> List[str]:
    """labels.yaml에서 schema_labels 목록을 반환."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["schema_labels"]
