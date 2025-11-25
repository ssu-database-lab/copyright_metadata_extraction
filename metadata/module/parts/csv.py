"""CSV 및 YAML 로드 유틸리티"""
import yaml
from pathlib import Path


def load_config(config_path='configs/labels.yaml'):
    """labels.yaml 설정 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_label_groups(config):
    """라벨 그룹 추출"""
    return {
        'regex': config.get('regex_labels', {}),
        'datetime': config.get('datetime_labels', []),
        'numeric': config.get('numeric_labels', []),
        'ner': config.get('ner_labels', []),
        'text': config.get('text_labels', [])
    }
