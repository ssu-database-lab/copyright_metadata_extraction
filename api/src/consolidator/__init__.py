#!/usr/bin/env python3
"""
Metadata Consolidation Module

This module provides intelligent consolidation of metadata extracted from
both LLM-based extraction and NER-based extraction systems.

Main Components:
- ConsolidationAgent: Main orchestrator using Qwen3-Next-80B
- FieldMapper: Maps NER entities to LLM metadata fields
- ValidationEngine: Validates formats and logic
- ReasoningGenerator: Generates evidence and reasoning
"""

from .consolidation_agent import ConsolidationAgent
from .field_mapper import FieldMapper
from .validation_engine import ValidationEngine
from .reasoning_generator import ReasoningGenerator

__all__ = [
    'ConsolidationAgent',
    'FieldMapper',
    'ValidationEngine',
    'ReasoningGenerator'
]

__version__ = '1.0.0'

