#!/usr/bin/env python3
"""
Consolidation Agent - Main orchestrator for metadata consolidation

This module provides the main ConsolidationAgent class that uses
Qwen3-Next-80B to intelligently compare and merge LLM and NER results.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from module.llm_extraction import LLMExtractionProcessor
from module.consolidator.field_mapper import FieldMapper
from module.consolidator.validation_engine import ValidationEngine
from module.consolidator.reasoning_generator import ReasoningGenerator
from module.consolidator.schemas.consolidation_schemas import ConsolidationSchemas

logger = logging.getLogger(__name__)


def _is_empty(v) -> bool:
    return v in (None, "", [], {})


def _norm_decisions(decisions) -> List[Dict[str, Any]]:
    """중재자의 decisions → dict 목록. {필드: {...}} 형태도 받고, 글자만 있는 항목은 버린다
    (예전에는 문자열 항목에서 'str' object has no attribute 'copy' 로 통합 전체가 실패했다)."""
    if isinstance(decisions, dict):
        decisions = [{"field": k, **v} if isinstance(v, dict) else {"field": k, "final_value": v}
                     for k, v in decisions.items()]
    if not isinstance(decisions, list):
        return []
    return [d for d in decisions if isinstance(d, dict)]


def _thinking_off(env_name: str) -> bool:
    """qwen3.5 의 생각(thinking) 모드를 끄는가 — 기본은 끔(2026-10-20 결정). 환경변수를 on 으로 주면 켠다.
    근거 (docs/v4/load_test_20261019.md): 계약서 200건(스캔 50) 끔 — 일치율 텍스트 99.9 %·스캔 97.2 %(켬 98.3–99.8 %·
    94.4–96.9 %), 신양식 5건 Macro-F1 100 %, 1건 16.7k 토큰(켬 ~24k), 중앙 32초(켬 77–127초)."""
    return os.getenv(env_name, "off").strip().lower() in ("off", "0", "false", "no")


def _thinking_kwargs(env_name: str) -> Dict[str, Any]:
    return {"extra_body": {"enable_thinking": False}} if _thinking_off(env_name) else {}


def consolidation_timeout_sec(fallback: bool = False) -> float:
    """중재 호출 1회의 시간 제한(초). CONSOLIDATION_TIMEOUT_SEC 가 있으면 그 값이다.
    없으면 생각 끔 90초 · 켬 240초. 생각 끔 정상 호출은 8–48초인데(200건) 약 10 %가 응답 없이 멈췄다가 SDK 의 재시도로
    ~15초에 답했다 — 240초를 기다리면 그 작업자가 4분을 논다. 폴백(qwen3.5-plus)은 생각 끔에서 재 보지 않았으므로 240초를 둔다."""
    v = os.getenv("CONSOLIDATION_TIMEOUT_SEC")
    if v:
        return float(v)
    return 90.0 if _thinking_off("CONSOLIDATION_THINKING") and not fallback else 240.0


def _nonnull_count(d) -> int:
    return sum(1 for v in (d or {}).values() if v not in (None, "", [], {}))


def _consolidation_diagnostics(llm_result, consolidated_result) -> Dict[str, Any]:
    """중재 전후를 나란히 남긴다 — 결과가 줄어든 경우 원인을 나중에 볼 수 있도록.

    통합이 성공을 보고하면서 필드를 통째로 잃는 사례가 관측됐는데(최근 60건 중 9건,
    대부분 저작물측 9필드 VLM 결과), 중재자 원문 응답을 저장하지 않아 왜 그런지
    확인할 수 없었다. 응답을 버리지 말고 남긴다. 손실이 없을 때는 응답 전문을
    보관하지 않는다 — 정상 건까지 쌓으면 결과 디렉터리가 불필요하게 커진다.
    """
    raw_md = (llm_result or {}).get("metadata") or {}
    con_md = (consolidated_result or {}).get("consolidated_metadata") or {}
    before, after = _nonnull_count(raw_md), _nonnull_count(con_md)
    removed = set((consolidated_result or {}).get("removed_by_arbiter") or [])
    lost = [k for k, v in raw_md.items()
            if v not in (None, "", [], {}) and con_md.get(k) in (None, "", [], {}) and k not in removed]
    diag = {
        "consolidation_fields_before": before,
        "consolidation_fields_after": after,
        "consolidation_lost_fields": lost,
        "consolidation_removed_fields": sorted(removed),
        "consolidation_kept_from_llm": (consolidated_result or {}).get("kept_from_llm") or [],
        "consolidation_truncated": bool((consolidated_result or {}).get("truncated")),
    }
    if removed:
        logger.info(f"중재자가 근거를 들어 비운 필드 {len(removed)}개: {sorted(removed)[:8]}")
    if lost:
        logger.error(
            f"통합에서 필드 손실: {before} → {after} (잃은 필드 {len(lost)}개: {lost[:8]}). "
            "중재자 원문 응답을 consolidation_raw_response 에 남깁니다.")
        diag["consolidation_raw_response"] = consolidated_result.get("raw_response")
        diag["consolidation_prompt_chars"] = consolidated_result.get("prompt_chars")
    return diag


class ConsolidationAgent:
    """
    Main consolidation agent using Qwen3-Next-80B
    
    This agent compares, validates, and merges results from both
    LLM extraction and NER extraction systems.
    """
    
    def __init__(
        self, 
        model_name: str = "alibaba-qwen3-next-80b-a3b-instruct",
        output_dir: Optional[str] = None,
        fallback_model: Optional[str] = "alibaba-qwen3.5-plus",
        enable_hybrid: bool = True
    ):
        """
        Initialize consolidation agent
        
        Args:
            model_name: Primary LLM model to use (default: Qwen3-Next-80B)
            output_dir: Output directory for results
            fallback_model: Fallback model if primary fails (default: Qwen-Max)
            enable_hybrid: Enable hybrid approach with automatic fallback
        """
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.enable_hybrid = enable_hybrid
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Initialize components
        self.llm_processor = LLMExtractionProcessor(output_dir=str(self.output_dir) if self.output_dir else None)
        self.field_mapper = FieldMapper()
        self.validator = ValidationEngine()
        self.reasoner = ReasoningGenerator()
        
        # Initialize primary LLM model
        if not self.llm_processor.initialize_model(model_name):
            raise RuntimeError(f"Failed to initialize model: {model_name}")
        
        logger.debug(f"중재자 준비 — 모델 {model_name}" + (f" · 폴백 {fallback_model}" if enable_hybrid and fallback_model else ""))
    
    def consolidate(
        self,
        llm_result: Dict[str, Any],
        ner_result: Dict[str, Any],
        ocr_text: str,
        document_type: str = "기타문서"
    ) -> Dict[str, Any]:
        """
        Consolidate LLM and NER results using Qwen3-Next-80B
        
        Args:
            llm_result: Result from LLM extraction
                Expected keys: 'metadata', 'confidence', 'success', etc.
            ner_result: Result from NER extraction
                Expected keys: 'extracted_entities', 'statistics', 'success', etc.
            ocr_text: Original OCR text
            document_type: Type of document (계약서, 동의서, etc.)
        
        Returns:
            Dictionary with consolidated metadata and validation report
        """
        try:
            logger.debug("통합 검증 시작")
            
            # Extract data
            llm_metadata = llm_result.get('metadata', {})
            ner_entities = ner_result.get('extracted_entities', [])
            
            if not llm_metadata and not ner_entities:
                logger.warning("통합 검증 — LLM·NER 결과가 모두 비어 있어 빈 결과를 냅니다")
                return self._create_empty_result()
            
            # Step 1: Field mapping (no LLM needed)
            logger.debug("1단계: NER 엔티티 → LLM 필드 대응")
            field_mappings = self.field_mapper.map_entities_to_fields(
                ner_entities=ner_entities,
                llm_metadata=llm_metadata,
                ocr_text=ocr_text,
                document_type=document_type
            )
            
            # Step 2: Validation (no LLM needed)
            logger.debug("2단계: 형식·논리 검증")
            validation_errors = self.validator.validate_logic(
                metadata=llm_metadata,
                document_type=document_type
            )
            
            # Step 3: LLM-based consolidation (with hybrid fallback)
            logger.debug(f"3단계: LLM 중재 (NER 엔티티 {len(ner_entities)}개 · 대응 필드 {len(field_mappings or {})}개 · "
                         f"검증 오류 {len(validation_errors or [])}건)")
            consolidated_result = self._llm_consolidate(
                llm_metadata=llm_metadata,
                ner_entities=ner_entities,
                field_mappings=field_mappings,
                ocr_text=ocr_text,
                document_type=document_type
            )
            
            if not consolidated_result.get('degraded'):
                consolidated_result = self._merge_with_llm(llm_metadata, consolidated_result, field_mappings)

            # Step 4: Post-process and generate evidence
            logger.debug("4단계: 근거·판정 이유 정리")
            final_result = self._post_process(
                consolidated_result=consolidated_result,
                llm_result=llm_result,
                ner_result=ner_result,
                field_mappings=field_mappings,
                validation_errors=validation_errors,
                ocr_text=ocr_text,
            )
            
            # Add model usage info to final result
            if consolidated_result.get('model_used'):
                final_result['model_used'] = consolidated_result['model_used']
                final_result['fallback_used'] = consolidated_result.get('fallback_used', False)
                if consolidated_result.get('primary_model_failed'):
                    final_result['primary_model_error'] = consolidated_result.get('primary_model_error')
            
            logger.debug("통합 검증 끝")
            return final_result
            
        except Exception as e:
            logger.error(f"통합 검증 오류 — {type(e).__name__}: {e}", exc_info=True)
            return self._create_error_result(str(e))
    
    def _llm_consolidate(
        self,
        llm_metadata: Dict[str, Any],
        ner_entities: List[tuple],
        field_mappings: Dict[str, List],
        ocr_text: str,
        document_type: str
    ) -> Dict[str, Any]:
        """
        Use LLM to consolidate metadata with hybrid fallback approach
        
        Tries primary model first, falls back to Qwen-Max if JSON parsing fails
        """
        # Create consolidation prompt text
        prompt_text = self._create_consolidation_prompt(
            llm_metadata=llm_metadata,
            ner_entities=ner_entities,
            field_mappings=field_mappings,
            ocr_text=ocr_text,
            document_type=document_type
        )
        
        # Get consolidation schema
        schema = ConsolidationSchemas.get_consolidation_schema()
        
        # Try primary model first
        logger.debug(f"주 중재자 호출 — {self.model_name} · 프롬프트 {len(prompt_text or ''):,}자")
        try:
            result = self._call_model_for_consolidation(
                model_name=self.model_name,
                prompt_text=prompt_text,
                schema=schema,
                document_type=document_type
            )
            
            # 읽은 결과가 쓸 만해야 성공 — LLM 에 값이 있는데 중재 결과가 비었으면 실패로 보고 폴백한다
            if self._usable(result, llm_metadata):
                result['model_used'] = self.model_name
                result['fallback_used'] = False
                logger.debug(f"주 중재자 성공 — {self.model_name}")
                return result
            else:
                raise ValueError("Failed to parse JSON from primary model response")
                
        except (ValueError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"주 중재자 실패 — {self.model_name}: {type(e).__name__}: {str(e)[:300]}"
                           + (f" → 폴백 {self.fallback_model}" if self.enable_hybrid and self.fallback_model
                              else " → 규칙 기반 병합(저하)"))
            
            # Try fallback model if hybrid mode is enabled
            if self.enable_hybrid and self.fallback_model:
                logger.info(f"폴백 중재자로 다시 시도 — {self.fallback_model}")
                try:
                    # Initialize fallback model
                    fallback_processor = LLMExtractionProcessor(
                        output_dir=str(self.output_dir) if self.output_dir else None
                    )
                    if not fallback_processor.initialize_model(self.fallback_model):
                        raise RuntimeError(f"Failed to initialize fallback model: {self.fallback_model}")
                    
                    result = self._call_model_for_consolidation(
                        model_name=self.fallback_model,
                        prompt_text=prompt_text,
                        schema=schema,
                        document_type=document_type,
                        extractor=fallback_processor.extractor
                    )
                    
                    if self._usable(result, llm_metadata):
                        result['model_used'] = self.fallback_model
                        result['fallback_used'] = True
                        result['primary_model_failed'] = True
                        result['primary_model_error'] = str(e)
                        logger.info(f"폴백 중재자 성공 — {self.fallback_model}")
                        return result
                    else:
                        raise ValueError("Failed to parse JSON from fallback model response")
                        
                except Exception as fallback_error:
                    logger.error(f"폴백 중재자도 실패 — {self.fallback_model}: {type(fallback_error).__name__}: "
                                 f"{str(fallback_error)[:300]} → 규칙 기반 병합(저하)")
                    # Both models failed, use basic fallback
                    return self._fallback_consolidate(
                        llm_metadata, ner_entities, field_mappings,
                        reason=f"주 모델({self.model_name}) 및 폴백({self.fallback_model}) 모두 실패: "
                               f"{type(fallback_error).__name__}: {str(fallback_error)[:120]}"
                    )
            else:
                # Hybrid mode disabled or no fallback model, use basic fallback
                return self._fallback_consolidate(
                    llm_metadata, ner_entities, field_mappings,
                    reason=f"주 모델({self.model_name}) 실패, 폴백 미설정: "
                           f"{type(e).__name__}: {str(e)[:120]}"
                )
    
    def _call_model_for_consolidation(
        self,
        model_name: str,
        prompt_text: str,
        schema: Dict[str, Any],
        document_type: str,
        extractor=None
    ) -> Dict[str, Any]:
        """
        Call a specific model for consolidation
        
        Args:
            model_name: Model to use
            prompt_text: Consolidation prompt
            schema: Consolidation schema
            document_type: Document type
            extractor: Optional extractor (if None, uses self.llm_processor.extractor)
        
        Returns:
            Dict with consolidated_metadata, decisions, summary, etc.
        """
        try:
            # Use provided extractor or get from processor
            if extractor is None:
                extractor = self.llm_processor.extractor
                if not extractor:
                    raise RuntimeError("LLM extractor not initialized")
            
            # Store raw response for debugging
            raw_response_text = None
            
            # Check if it's a cloud extractor wrapper (for Alibaba Cloud models)
            from module.llm_extraction.models.base_extractor import CloudExtractorWrapper
            
            if isinstance(extractor, CloudExtractorWrapper):
                # For cloud models, we can use the consolidation prompt directly
                # The CloudExtractorWrapper wraps the actual cloud extractor
                cloud_extractor = extractor.cloud_extractor
                
                # Create messages with our custom consolidation prompt
                messages = [
                    {
                        "role": "system",
                        "content": "You are a Korean document metadata consolidation expert. "
                                   "Your task is to compare and merge results from two extraction methods (LLM and NER). "
                                   "Output ONLY valid JSON. No markdown, no code blocks, no explanations. "
                                   "Ensure all braces, brackets, and quotes are properly closed. "
                                   "Escape special characters in strings (\\\" \\n). "
                                   "Maintain complete JSON structure even for long responses."
                    },
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ]

                # Call the cloud extractor's API directly
                schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

                # For Alibaba Cloud, call the API
                if hasattr(cloud_extractor, 'client'):
                    try:
                        _t_call = datetime.now()
                        response = cloud_extractor.client.chat.completions.create(
                            model=cloud_extractor.dashscope_model_id,
                            messages=messages,
                            temperature=0.1,   # Low temperature for deterministic JSON
                            top_p=0.8,
                            max_tokens=8192,
                            # 생각 켬 240초는 실측에서 나왔다(2026-09, 30계약서 × 6중재자 × 2회):
                            #   현행 qwen3.5-122b  중앙 68초 · p95 108초
                            #   폴백 qwen3.5-plus  중앙 149초 · p95 180초
                            # 그때 90초에서는 폴백이 **단 한 번도 성공한 적이 없었다**. 생각 끔(기본)은 주 모델만 90초
                            # — consolidation_timeout_sec() 참고.
                            timeout=consolidation_timeout_sec(fallback=model_name == self.fallback_model),
                            response_format={"type": "json_object"},  # Force valid JSON output
                            **_thinking_kwargs("CONSOLIDATION_THINKING"),
                        )
                        
                        extracted_text = response.choices[0].message.content
                        raw_response_text = extracted_text
                        
                        # Check if response was truncated
                        finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
                        _usage = getattr(response, "usage", None)
                        _tok = (f" · 토큰 입력 {_usage.prompt_tokens:,} / 출력 {_usage.completion_tokens:,}"
                                if _usage is not None and isinstance(getattr(_usage, "prompt_tokens", None), int)
                                and isinstance(getattr(_usage, "completion_tokens", None), int) else "")
                        logger.info(f"중재자 응답 — 모델 {cloud_extractor.dashscope_model_id} · "
                                    f"{(datetime.now() - _t_call).total_seconds():.1f}초{_tok} · 종료 사유 {finish_reason} · "
                                    f"프롬프트 {len(prompt_text or ''):,}자")
                        if finish_reason == 'length':
                            logger.warning(f"중재자 응답이 길이 제한(max_tokens)에서 잘렸습니다 (finish_reason: {finish_reason}) — JSON 이 불완전할 수 있습니다")
                            # Try to extract what we have, but note it may be incomplete
                        
                        # Parse JSON response with improved error handling
                        cleaned_text = cloud_extractor._clean_markdown_formatting(extracted_text) if hasattr(cloud_extractor, '_clean_markdown_formatting') else extracted_text
                        
                        consolidated_metadata = None
                        json_parse_errors = []
                        
                        # Strategy 1: Try direct JSON parse
                        try:
                            consolidated_metadata = json.loads(cleaned_text)
                            logger.debug("중재자 JSON 해석 성공")
                        except json.JSONDecodeError as json_err:
                            json_parse_errors.append(f"Direct parse failed: {json_err}")
                            logger.warning(f"중재자 JSON 해석 실패 — 다른 방법으로 복구를 시도합니다: {json_err}")
                            
                            # Strategy 2: Try to extract JSON from markdown code blocks
                            import re
                            # Look for JSON in code blocks first
                            json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                            json_block_match = re.search(json_block_pattern, cleaned_text, re.DOTALL)
                            if json_block_match:
                                try:
                                    consolidated_metadata = json.loads(json_block_match.group(1))
                                    logger.info("중재자 JSON 복구 — 코드 블록에서 읽음")
                                except json.JSONDecodeError as e:
                                    json_parse_errors.append(f"Code block parse failed: {e}")
                            
                            # Strategy 3: Try to find JSON object boundaries more carefully
                            if consolidated_metadata is None:
                                # Find the first { and try to match braces
                                start_idx = cleaned_text.find('{')
                                if start_idx != -1:
                                    brace_count = 0
                                    in_string = False
                                    escape_next = False
                                    end_idx = start_idx
                                    
                                    for i in range(start_idx, len(cleaned_text)):
                                        char = cleaned_text[i]
                                        
                                        if escape_next:
                                            escape_next = False
                                            continue
                                        
                                        if char == '\\':
                                            escape_next = True
                                            continue
                                        
                                        if char == '"' and not escape_next:
                                            in_string = not in_string
                                            continue
                                        
                                        if not in_string:
                                            if char == '{':
                                                brace_count += 1
                                            elif char == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    end_idx = i + 1
                                                    break
                                    
                                    if brace_count == 0 and end_idx > start_idx:
                                        json_str = cleaned_text[start_idx:end_idx]
                                        try:
                                            consolidated_metadata = json.loads(json_str)
                                            logger.info("중재자 JSON 복구 — 중괄호 짝 맞추기로 읽음")
                                        except json.JSONDecodeError as e:
                                            json_parse_errors.append(f"Brace matching parse failed: {e}")
                            
                            # Strategy 4: Try to repair incomplete JSON (if cut off)
                            if consolidated_metadata is None:
                                # Try to find the last complete JSON object
                                last_brace = cleaned_text.rfind('}')
                                if last_brace != -1:
                                    # Try to extract up to the last complete brace
                                    potential_json = cleaned_text[:last_brace + 1]
                                    # Try to close any unclosed strings or structures
                                    try:
                                        # Simple repair: close unclosed strings
                                        if potential_json.count('"') % 2 != 0:
                                            potential_json += '"'
                                        if potential_json.count('{') > potential_json.count('}'):
                                            potential_json += '}' * (potential_json.count('{') - potential_json.count('}'))
                                        
                                        consolidated_metadata = json.loads(potential_json)
                                        logger.info("중재자 JSON 복구 — 수리 후 읽음")
                                    except json.JSONDecodeError as e:
                                        json_parse_errors.append(f"Repair attempt failed: {e}")
                            
                            # Strategy 5: Last resort - try regex extraction (non-greedy)
                            if consolidated_metadata is None:
                                # Use non-greedy matching to find the first complete JSON object
                                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
                                if json_match:
                                    try:
                                        consolidated_metadata = json.loads(json_match.group())
                                        logger.info("중재자 JSON 복구 — 정규식 추출로 읽음")
                                    except json.JSONDecodeError as e:
                                        json_parse_errors.append(f"Regex extraction failed: {e}")
                            
                            # If all strategies failed, raise error with details
                            if consolidated_metadata is None:
                                error_msg = f"Could not parse JSON from response after {len(json_parse_errors)} attempts.\n"
                                error_msg += f"Errors: {'; '.join(json_parse_errors)}\n"
                                error_msg += f"Response preview (first 500 chars): {cleaned_text[:500]}\n"
                                error_msg += f"Response length: {len(cleaned_text)} chars"
                                
                                # Save raw response for debugging
                                if self.output_dir:
                                    debug_file = self.output_dir / f"consolidation_debug_response_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                    with open(debug_file, 'w', encoding='utf-8') as f:
                                        f.write(f"Model: {model_name}\n")
                                        f.write(f"Raw Response:\n{raw_response_text}\n\n")
                                        f.write(f"Cleaned Text:\n{cleaned_text}\n\n")
                                        f.write(f"Parse Errors:\n{chr(10).join(json_parse_errors)}")
                                    logger.error(f"중재자 JSON 을 읽지 못해 원문 응답을 저장했습니다: {debug_file.name}")
                                
                                raise ValueError(error_msg)
                        
                        # 중재자 응답에서 결과를 꺼낸다. 감싼 형태와 평평한 형태를 모두 받는다.
                        #
                        # 원래 코드는 평평한 형태를 처리하려 했지만 그 else 가지가
                        # 실행될 수 없었다 — if 에서 이미 dict 인지 보고, else 안에서
                        # 다시 isinstance(dict) 를 확인했기 때문이다. dict 가 아니면
                        # else 로 오지만 그 안의 조건도 전부 거짓이 된다. 결국 모델이
                        # {"work_title": ...} 처럼 평평하게 답하면 .get('consolidated_metadata')
                        # 가 {} 를 돌려주고, 추출 결과가 통째로 사라진 채
                        # status="completed" 로 저장됐다.
                        decisions, summary, final_metadata = [], {}, {}
                        if isinstance(consolidated_metadata, dict):
                            decisions = consolidated_metadata.get('decisions', []) or []
                            summary = consolidated_metadata.get('summary', {}) or {}
                            final_metadata = consolidated_metadata.get('consolidated_metadata') or {}
                            if not final_metadata:
                                # 감싸는 키가 없다 = 평평한 형태. 제어용 키를 뺀 나머지가 결과다.
                                flat = {k: v for k, v in consolidated_metadata.items()
                                        if k not in ('decisions', 'summary', 'consolidated_metadata')}
                                if flat:
                                    logger.warning(
                                        "중재자가 consolidated_metadata 래퍼 없이 응답했습니다 — "
                                        f"루트의 {len(flat)}개 필드를 결과로 사용합니다.")
                                    final_metadata = flat
                        
                        logger.debug(f"중재자 판정 {len(decisions)}건")
                        
                        # Calculate confidence from decisions if summary doesn't have it
                        if summary and 'overall_confidence' in summary:
                            llm_confidence = summary['overall_confidence']
                        elif decisions:
                            # Calculate average confidence from decisions
                            confidences = [d.get('confidence', 0.0) for d in decisions if d.get('confidence')]
                            llm_confidence = sum(confidences) / len(confidences) if confidences else 0.7
                        else:
                            llm_confidence = 0.7
                        
                        return {
                            "consolidated_metadata": final_metadata,
                            "decisions": decisions,
                            "summary": summary,
                            "status": "completed",
                            "llm_confidence": llm_confidence,
                            "raw_response": raw_response_text,
                            "prompt_chars": len(prompt_text or ""),
                            "truncated": finish_reason == "length",
                        }
                        
                    except Exception as e:
                        logger.warning(f"중재자 API 호출 실패 — {model_name}: {type(e).__name__}: {str(e)[:300]}")
                        raise  # Re-raise to trigger fallback
            else:
                # Not a cloud extractor, use standard extract_metadata
                # 프롬프트를 자르지 않는다. 예전에는 [:2000] 으로 잘랐는데, 실제
                # 통합검증 프롬프트는 저작물 9필드에서 2,074자·계약서 44필드에서 8,383자다.
                # 잘린 자리에 정확히 "Required JSON output format" 블록과 마지막
                # "JSON response:" 지시가 들어 있어서, 모델은 출력 형식을 모른 채 답하고
                # 그 결과 consolidated_metadata 래퍼가 없는 응답이 나왔다.
                result = extractor.extract_metadata(
                    text=prompt_text,
                    schema=schema,
                    document_type=f"{document_type}_consolidation"
                )
                from module.llm_extraction.models.base_extractor import ExtractionResult
                if isinstance(result, ExtractionResult):
                    consolidated_metadata = result.metadata
                else:
                    consolidated_metadata = result
                
                # 위와 같은 규칙 — 래퍼가 없으면 루트를 결과로 본다.
                decisions, summary, final_metadata = [], {}, {}
                if isinstance(consolidated_metadata, dict):
                    decisions = consolidated_metadata.get('decisions', []) or []
                    summary = consolidated_metadata.get('summary', {}) or {}
                    final_metadata = consolidated_metadata.get('consolidated_metadata') or {}
                    if not final_metadata:
                        flat = {k: v for k, v in consolidated_metadata.items()
                                if k not in ('decisions', 'summary', 'consolidated_metadata')}
                        if flat:
                            logger.warning("중재자 응답에 래퍼가 없어 루트 "
                                           f"{len(flat)}개 필드를 결과로 사용합니다.")
                            final_metadata = flat
                
                # Calculate confidence
                if summary and 'overall_confidence' in summary:
                    llm_confidence = summary['overall_confidence']
                elif decisions:
                    confidences = [d.get('confidence', 0.0) for d in decisions if d.get('confidence')]
                    llm_confidence = sum(confidences) / len(confidences) if confidences else 0.7
                else:
                    llm_confidence = 0.7
                
                return {
                    "consolidated_metadata": final_metadata,
                    "decisions": decisions,
                    "summary": summary,
                    "status": "completed",
                    "llm_confidence": llm_confidence,
                    "raw_response": None,   # 이 경로는 원문 텍스트를 보관하지 않는다
                    "prompt_chars": len(prompt_text or "")
                }
        except Exception as e:
            logger.debug(f"중재자 호출 오류 상세 — {model_name}: {e}", exc_info=True)
            raise  # Re-raise to allow fallback handling
    
    def _create_consolidation_prompt(
        self,
        llm_metadata: Dict[str, Any],
        ner_entities: List[tuple],
        field_mappings: Dict[str, List],
        ocr_text: str,
        document_type: str
    ) -> str:
        """중재 프롬프트 — **값이 있는 필드만** 보내고, 값이 있는 필드만 답하게 한다.

        예전에는 87개 필드(대부분 빈 값)를 모두 보내고 '빈 값까지 모든 필드를, 필드마다 판정 이유와 함께' 답하게 했다.
        응답이 8,192 토큰 상한에 닿아 잘리거나(계약서 336건 중 2건, 이미지 8건 중 6건) 비어 돌아왔다 (2026-10-19/20).
        빠진 필드는 _merge_with_llm 이 LLM 값으로 지키므로, 중재자는 판단이 필요한 필드만 다루면 된다.
        """
        ner_by_type: Dict[str, List[str]] = {}
        for entity_text, entity_type in ner_entities:
            ner_by_type.setdefault(entity_type, []).append(entity_text)
        present = {k: v for k, v in (llm_metadata or {}).items() if not _is_empty(v)}
        mapped = {k: v for k, v in (field_mappings or {}).items() if v}

        # Truncate OCR text if too long (keep first 3000 chars for context)
        ocr_excerpt = ocr_text[:3000] + ("..." if len(ocr_text) > 3000 else "")

        prompt = f"""Compare and merge the LLM and NER extraction results for this Korean document.
Output ONLY valid JSON. No markdown, no explanations.

## Original OCR text:
{ocr_excerpt}

## LLM extraction result — only the {len(present)} fields that have a value (all other fields are empty):
{json.dumps(present, ensure_ascii=False, indent=1)}

## NER extraction result (entity list):
{json.dumps(ner_by_type, ensure_ascii=False, indent=1)}

## Field mapping (NER entities mapped to LLM fields):
{json.dumps(mapped, ensure_ascii=False, indent=1)}

## Instructions:

1. Look only at fields that have an LLM value or a mapped NER entity. Never list empty fields.
2. Decide each such field:
   - Both match → "AGREED"
   - Both exist but differ → "CONFLICT" (check the OCR text and pick the correct one)
   - Only in LLM → "LLM_ONLY"
   - Only in NER → "NER_ONLY"
   - The OCR text shows the value is wrong and no correct value exists → "REMOVED" (final_value null)
3. consolidated_metadata: only the fields whose final value is not null. A field you leave out keeps its LLM value.
4. decisions is required: one entry for every field you looked at, not only conflicts.
   reasoning: one short Korean sentence (60 characters or fewer).
5. confidence: AGREED 0.9-1.0 · CONFLICT 0.7-0.9 · LLM_ONLY 0.5-0.7 · NER_ONLY 0.6-0.8 · REMOVED 0.0

## Required JSON output format:

{{
  "consolidated_metadata": {{"field_name": "final value"}},
  "decisions": [
    {{"field": "field_name", "llm_value": "…", "ner_value": "…", "final_value": "…",
      "decision": "AGREED", "reasoning": "짧은 한국어 이유", "confidence": 1.0}}
  ],
  "summary": {{"total_fields": 0, "agreed_fields": 0, "conflicted_fields": 0, "llm_only_fields": 0,
              "ner_only_fields": 0, "missing_fields": 0, "overall_confidence": 0.9}}
}}

JSON response:
"""
        return prompt
    
    @staticmethod
    def _usable(result: Optional[Dict[str, Any]], llm_metadata: Dict[str, Any]) -> bool:
        """중재 결과를 쓸 수 있는가 — 결과가 있고, LLM 에 값이 있었다면 중재 결과도 비어 있지 않아야 한다."""
        if not result or result.get('consolidated_metadata') is None:
            return False
        if any(not _is_empty(v) for v in (llm_metadata or {}).values()):
            return bool(result.get('consolidated_metadata'))
        return True

    @staticmethod
    def _merge_with_llm(llm_metadata: Dict[str, Any], result: Dict[str, Any],
                        field_mappings: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """중재자 응답 → 완전한 결과.

        LLM 결과(전체 스키마)에서 시작해 중재자가 값을 준 필드를 덮어쓴다. 중재자가 빼먹은 필드는 LLM 값을 지키고,
        근거를 들어 REMOVED 로 판정한 필드만 비운다. 스키마에 없는 키(지어낸 키)는 NER 대응 필드가 아니면 버린다.
        """
        base = dict(llm_metadata or {})
        arb = result.get('consolidated_metadata') or {}
        allowed = set(base) | set((field_mappings or {}).keys())
        removed = {d.get('field') for d in _norm_decisions(result.get('decisions'))
                   if str(d.get('decision', '')).upper() == 'REMOVED' and d.get('field')}
        for k, v in arb.items():
            if _is_empty(v) or (allowed and k not in allowed):
                continue
            base[k] = v
        for k in removed:
            if k in base:
                base[k] = None
        kept = [k for k, v in (llm_metadata or {}).items()
                if not _is_empty(v) and _is_empty(arb.get(k)) and k not in removed]
        return {**result, 'consolidated_metadata': base, 'kept_from_llm': kept,
                'removed_by_arbiter': sorted(removed & set(base))}

    def _post_process(
        self,
        consolidated_result: Dict[str, Any],
        llm_result: Dict[str, Any],
        ner_result: Dict[str, Any],
        field_mappings: Dict[str, List],
        validation_errors: List[str],
        ocr_text: str = "",
    ) -> Dict[str, Any]:
        """Post-process consolidation result"""
        
        decisions = _norm_decisions(consolidated_result.get('decisions', []))
        summary = consolidated_result.get('summary', {})
        
        # Calculate statistics from decisions if summary not provided
        if not summary:
            summary = {
                "total_fields": len(decisions),
                "agreed_fields": sum(1 for d in decisions if d.get('decision') == 'AGREED'),
                "conflicted_fields": sum(1 for d in decisions if d.get('decision') == 'CONFLICT'),
                "llm_only_fields": sum(1 for d in decisions if d.get('decision') == 'LLM_ONLY'),
                "ner_only_fields": sum(1 for d in decisions if d.get('decision') == 'NER_ONLY'),
                "missing_fields": sum(1 for d in decisions if d.get('decision') == 'MISSING'),
                "overall_confidence": consolidated_result.get('llm_confidence', 0.0)
            }
        
        # Enhance decisions with evidence
        enhanced_decisions = []
        for decision in decisions:
            enhanced_decision = decision.copy()
            
            # Add evidence using reasoning generator
            evidence = self.reasoner.generate_evidence(
                field_name=decision.get('field', ''),
                llm_value=decision.get('llm_value'),
                ner_value=decision.get('ner_value'),
                final_value=decision.get('final_value'),
                decision=decision.get('decision', 'UNKNOWN'),
                ocr_text=ocr_text or llm_result.get('ocr_text', ''),
                confidence=decision.get('confidence', 0.0)
            )
            
            # generate_evidence 는 평평한 dict 를 돌려준다 — 예전 코드는 없는 'evidence' 키를 읽어 늘 {} 였다
            enhanced_decision['evidence'] = {k: evidence.get(k) for k in ("reasoning", "ocr_excerpt")
                                             if evidence.get(k)}
            enhanced_decisions.append(enhanced_decision)
        
        # Generate validation report
        validation_report = {
            "confidence_score": summary.get('overall_confidence', consolidated_result.get('llm_confidence', 0.0)),
            "total_fields": summary.get('total_fields', len(consolidated_result.get('consolidated_metadata', {}))),
            "agreed_fields": summary.get('agreed_fields', 0),
            "conflicted_fields": summary.get('conflicted_fields', 0),
            "llm_only_fields": summary.get('llm_only_fields', 0),
            "ner_only_fields": summary.get('ner_only_fields', 0),
            "missing_fields": summary.get('missing_fields', 0),
            "validation_errors": validation_errors,
            "decisions": enhanced_decisions,
            "summary": summary
        }
        
        # LLM 중재가 실제로 일어났는지. 저하된 결과를 정상 결과처럼 저장하면
        # 감사에서 구분할 방법이 없다.
        degraded = bool(consolidated_result.get('degraded'))
        return {
            "success": True,
            "consolidated_metadata": consolidated_result.get('consolidated_metadata', {}),
            "validation_report": validation_report,
            "llm_metadata": llm_result.get('metadata', {}),
            "ner_entities": ner_result.get('extracted_entities', []),
            # 중재가 없었으면 요청 모델명을 적지 않는다 — 그 모델은 아무것도 만들지 않았다
            "model_used": ("(규칙 기반 폴백 — LLM 중재 없음)" if degraded
                           else consolidated_result.get('model_used') or self.model_name),
            "requested_model": self.model_name,
            "consolidation_degraded": degraded,
            "degraded_reason": consolidated_result.get('degraded_reason'),
            "status": consolidated_result.get('status', 'completed'),
            "llm_confidence": consolidated_result.get('llm_confidence', 0.0),
            **_consolidation_diagnostics(llm_result, consolidated_result),
        }
    
    def _fallback_consolidate(
        self,
        llm_metadata: Dict[str, Any],
        ner_entities: List[tuple],
        field_mappings: Dict[str, List],
        reason: str = "unknown"
    ) -> Dict[str, Any]:
        """LLM 중재가 실패했을 때의 규칙 기반 병합.

        **이건 성공이 아니다.** 통합검증의 핵심(LLM 중재)이 일어나지 않았고
        결과는 LLM 추출값을 거의 그대로 통과시킨 것에 가깝다. 호출자와 저장 결과가
        그 사실을 알 수 있어야 한다 — 예전에는 overall_confidence 를 0.6 으로 채워
        정상 결과와 구분되지 않았고, 그래서 폴백 중재자가 전 호출 실패 중인 것을
        아무도 알아채지 못했다(2026-09 발견).
        """
        logger.error(f"통합검증 LLM 중재 실패 → 규칙 기반 폴백으로 저하됨. 사유: {reason}")
        
        consolidated = llm_metadata.copy()
        decisions = []
        
        # Create decisions based on field mappings
        for field, entities in field_mappings.items():
            llm_value = self._get_nested_value(consolidated, field)
            ner_value = entities[0][0] if entities else None
            
            # Initialize default values
            decision = "MISSING"
            final_value = None
            confidence = 0.0
            explanation = "LLM과 NER 모두에서 해당 필드를 찾을 수 없음"
            
            if entities and llm_value:
                # Both present - check consistency
                ner_value = entities[0][0]
                is_consistent, conf, expl = self.validator.check_consistency(
                    llm_value, ner_value
                )
                if is_consistent:
                    decision = "AGREED"
                    final_value = llm_value
                    confidence = conf
                    explanation = expl
                else:
                    decision = "CONFLICT"
                    final_value = ner_value if conf > 0.6 else llm_value
                    confidence = conf
                    explanation = f"값이 일치하지 않음: {expl}"
            elif entities and not llm_value:
                # NER only
                ner_value = entities[0][0]
                decision = "NER_ONLY"
                final_value = ner_value
                confidence = 0.6
                explanation = "LLM에서 해당 필드를 찾을 수 없어 NER 값 사용"
            elif not entities and llm_value:
                # LLM only
                decision = "LLM_ONLY"
                final_value = llm_value
                confidence = 0.6
                explanation = "NER에서 해당 필드를 찾을 수 없어 LLM 값 사용"
            else:
                # Both missing
                decision = "MISSING"
                final_value = None
                confidence = 0.0
                explanation = "LLM과 NER 모두에서 해당 필드를 찾을 수 없음"
            
            decisions.append({
                "field": field,
                "llm_value": llm_value,
                "ner_value": ner_value,
                "final_value": final_value,
                "decision": decision,
                "reasoning": explanation,
                "confidence": confidence
            })
        
        return {
            "consolidated_metadata": consolidated,
            "decisions": decisions,
            "summary": {
                "total_fields": len(decisions),
                "agreed_fields": sum(1 for d in decisions if d.get('decision') == 'AGREED'),
                "conflicted_fields": sum(1 for d in decisions if d.get('decision') == 'CONFLICT'),
                "llm_only_fields": sum(1 for d in decisions if d.get('decision') == 'LLM_ONLY'),
                "ner_only_fields": sum(1 for d in decisions if d.get('decision') == 'NER_ONLY'),
                "missing_fields": sum(1 for d in decisions if d.get('decision') == 'MISSING'),
                # 0.6 은 '조금 낮은 정상값'처럼 보여서 필터를 통과했다. LLM 중재가
                # 아예 없었으므로 통합 신뢰도는 0 이다.
                "overall_confidence": 0.0,
                "degraded": True,
                "degraded_reason": reason
            },
            "status": "fallback",
            "degraded": True,
            "degraded_reason": reason
        }
    
    def _get_nested_value(self, metadata: Dict[str, Any], field_path: str) -> Optional[Any]:
        """Get value from nested dictionary using dot notation or array notation"""
        if '[]' in field_path:
            base_field, sub_field = field_path.split('[].')
            if base_field in metadata and isinstance(metadata[base_field], list):
                for item in metadata[base_field]:
                    if isinstance(item, dict) and sub_field in item:
                        return item[sub_field]
            return None
        
        parts = field_path.split('.')
        value = metadata
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result when inputs are empty"""
        return {
            "success": False,
            "error": "Both LLM and NER results are empty",
            "consolidated_metadata": {},
            "validation_report": {
                "confidence_score": 0.0,
                "total_fields": 0,
                "agreed_fields": 0,
                "conflicted_fields": 0,
                "llm_only_fields": 0,
                "ner_only_fields": 0,
                "validation_errors": [],
                "decisions": []
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "success": False,
            "error": error_message,
            "consolidated_metadata": {},
            "validation_report": {
                "confidence_score": 0.0,
                "total_fields": 0,
                "validation_errors": [error_message],
                "decisions": []
            }
        }

