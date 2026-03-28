#!/usr/bin/env python3
"""
Consolidation Agent - Main orchestrator for metadata consolidation

This module provides the main ConsolidationAgent class that uses
Qwen3-Next-80B to intelligently compare and merge LLM and NER results.
"""

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
        
        logger.info(f"ConsolidationAgent initialized with model: {model_name}")
        if enable_hybrid and fallback_model:
            logger.info(f"Hybrid mode enabled - fallback model: {fallback_model}")
    
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
            logger.info("Starting consolidation process")
            
            # Extract data
            llm_metadata = llm_result.get('metadata', {})
            ner_entities = ner_result.get('extracted_entities', [])
            
            if not llm_metadata and not ner_entities:
                logger.warning("Both LLM and NER results are empty")
                return self._create_empty_result()
            
            # Step 1: Field mapping (no LLM needed)
            logger.info("Step 1: Mapping NER entities to LLM fields")
            field_mappings = self.field_mapper.map_entities_to_fields(
                ner_entities=ner_entities,
                llm_metadata=llm_metadata,
                ocr_text=ocr_text,
                document_type=document_type
            )
            
            # Step 2: Validation (no LLM needed)
            logger.info("Step 2: Validating formats and logic")
            validation_errors = self.validator.validate_logic(
                metadata=llm_metadata,
                document_type=document_type
            )
            
            # Step 3: LLM-based consolidation (with hybrid fallback)
            logger.info("Step 3: LLM consolidation (hybrid mode enabled)")
            consolidated_result = self._llm_consolidate(
                llm_metadata=llm_metadata,
                ner_entities=ner_entities,
                field_mappings=field_mappings,
                ocr_text=ocr_text,
                document_type=document_type
            )
            
            # Step 4: Post-process and generate evidence
            logger.info("Step 4: Generating evidence and reasoning")
            final_result = self._post_process(
                consolidated_result=consolidated_result,
                llm_result=llm_result,
                ner_result=ner_result,
                field_mappings=field_mappings,
                validation_errors=validation_errors
            )
            
            # Add model usage info to final result
            if consolidated_result.get('model_used'):
                final_result['model_used'] = consolidated_result['model_used']
                final_result['fallback_used'] = consolidated_result.get('fallback_used', False)
                if consolidated_result.get('primary_model_failed'):
                    final_result['primary_model_error'] = consolidated_result.get('primary_model_error')
            
            logger.info("Consolidation completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Error during consolidation: {e}", exc_info=True)
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
        logger.info(f"Attempting consolidation with primary model: {self.model_name}")
        try:
            result = self._call_model_for_consolidation(
                model_name=self.model_name,
                prompt_text=prompt_text,
                schema=schema,
                document_type=document_type
            )
            
            # Validate JSON was parsed successfully
            if result and result.get('consolidated_metadata') is not None:
                result['model_used'] = self.model_name
                result['fallback_used'] = False
                logger.info(f"✅ Successfully consolidated with {self.model_name}")
                return result
            else:
                raise ValueError("Failed to parse JSON from primary model response")
                
        except (ValueError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Primary model ({self.model_name}) failed: {e}")
            
            # Try fallback model if hybrid mode is enabled
            if self.enable_hybrid and self.fallback_model:
                logger.info(f"🔄 Falling back to {self.fallback_model} for better JSON quality")
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
                    
                    if result and result.get('consolidated_metadata') is not None:
                        result['model_used'] = self.fallback_model
                        result['fallback_used'] = True
                        result['primary_model_failed'] = True
                        result['primary_model_error'] = str(e)
                        logger.info(f"✅ Successfully consolidated with fallback model: {self.fallback_model}")
                        return result
                    else:
                        raise ValueError("Failed to parse JSON from fallback model response")
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback model ({self.fallback_model}) also failed: {fallback_error}")
                    # Both models failed, use basic fallback
                    return self._fallback_consolidate(
                        llm_metadata, ner_entities, field_mappings
                    )
            else:
                # Hybrid mode disabled or no fallback model, use basic fallback
                logger.warning("Hybrid mode disabled or no fallback model, using basic consolidation")
                return self._fallback_consolidate(
                    llm_metadata, ner_entities, field_mappings
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
                        response = cloud_extractor.client.chat.completions.create(
                            model=cloud_extractor.dashscope_model_id,
                            messages=messages,
                            temperature=0.1,   # Low temperature for deterministic JSON
                            top_p=0.8,
                            max_tokens=8192,
                            timeout=90.0,
                            response_format={"type": "json_object"}  # Force valid JSON output
                        )
                        
                        extracted_text = response.choices[0].message.content
                        raw_response_text = extracted_text
                        
                        # Check if response was truncated
                        finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
                        if finish_reason == 'length':
                            logger.warning(f"Response was truncated (finish_reason: {finish_reason}). Consider increasing max_tokens.")
                            # Try to extract what we have, but note it may be incomplete
                        
                        # Parse JSON response with improved error handling
                        cleaned_text = cloud_extractor._clean_markdown_formatting(extracted_text) if hasattr(cloud_extractor, '_clean_markdown_formatting') else extracted_text
                        
                        consolidated_metadata = None
                        json_parse_errors = []
                        
                        # Strategy 1: Try direct JSON parse
                        try:
                            consolidated_metadata = json.loads(cleaned_text)
                            logger.info("Successfully parsed JSON directly")
                        except json.JSONDecodeError as json_err:
                            json_parse_errors.append(f"Direct parse failed: {json_err}")
                            logger.warning(f"JSON parse error, trying alternative methods: {json_err}")
                            
                            # Strategy 2: Try to extract JSON from markdown code blocks
                            import re
                            # Look for JSON in code blocks first
                            json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
                            json_block_match = re.search(json_block_pattern, cleaned_text, re.DOTALL)
                            if json_block_match:
                                try:
                                    consolidated_metadata = json.loads(json_block_match.group(1))
                                    logger.info("Successfully parsed JSON from code block")
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
                                            logger.info("Successfully parsed JSON using brace matching")
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
                                        logger.info("Successfully parsed JSON after repair attempt")
                                    except json.JSONDecodeError as e:
                                        json_parse_errors.append(f"Repair attempt failed: {e}")
                            
                            # Strategy 5: Last resort - try regex extraction (non-greedy)
                            if consolidated_metadata is None:
                                # Use non-greedy matching to find the first complete JSON object
                                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_text, re.DOTALL)
                                if json_match:
                                    try:
                                        consolidated_metadata = json.loads(json_match.group())
                                        logger.info("Successfully parsed JSON using regex extraction")
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
                                    logger.error(f"Saved debug response to: {debug_file}")
                                
                                raise ValueError(error_msg)
                        
                        # Extract decisions from consolidated metadata
                        # Handle both nested and flat structures
                        if isinstance(consolidated_metadata, dict):
                            decisions = consolidated_metadata.get('decisions', [])
                            summary = consolidated_metadata.get('summary', {})
                            final_metadata = consolidated_metadata.get('consolidated_metadata', {})
                        else:
                            # If response is flat, try to extract from root
                            decisions = consolidated_metadata.get('decisions', []) if isinstance(consolidated_metadata, dict) else []
                            summary = consolidated_metadata.get('summary', {}) if isinstance(consolidated_metadata, dict) else {}
                            final_metadata = consolidated_metadata if isinstance(consolidated_metadata, dict) else {}
                        
                        logger.info(f"Consolidation complete: {len(decisions)} decisions made")
                        
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
                            "raw_response": raw_response_text
                        }
                        
                    except Exception as e:
                        logger.error(f"Cloud API call failed for {model_name}: {e}")
                        raise  # Re-raise to trigger fallback
            else:
                # Not a cloud extractor, use standard extract_metadata
                result = extractor.extract_metadata(
                    text=prompt_text[:2000],  # Truncate if too long
                    schema=schema,
                    document_type=f"{document_type}_consolidation"
                )
                from module.llm_extraction.models.base_extractor import ExtractionResult
                if isinstance(result, ExtractionResult):
                    consolidated_metadata = result.metadata
                else:
                    consolidated_metadata = result
                
                # Extract decisions from consolidated metadata
                if isinstance(consolidated_metadata, dict):
                    decisions = consolidated_metadata.get('decisions', [])
                    summary = consolidated_metadata.get('summary', {})
                    final_metadata = consolidated_metadata.get('consolidated_metadata', {})
                else:
                    decisions = []
                    summary = {}
                    final_metadata = {}
                
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
                    "raw_response": None
                }
        except Exception as e:
            logger.error(f"Error calling model {model_name} for consolidation: {e}", exc_info=True)
            raise  # Re-raise to allow fallback handling
    
    def _create_consolidation_prompt(
        self,
        llm_metadata: Dict[str, Any],
        ner_entities: List[tuple],
        field_mappings: Dict[str, List],
        ocr_text: str,
        document_type: str
    ) -> str:
        """
        Create consolidation prompt for Qwen3-Next-80B
        
        Creates a comprehensive prompt that instructs the LLM to compare
        and merge LLM and NER extraction results.
        """
        # Prepare NER entities grouped by type
        ner_by_type = {}
        for entity_text, entity_type in ner_entities:
            if entity_type not in ner_by_type:
                ner_by_type[entity_type] = []
            ner_by_type[entity_type].append(entity_text)
        
        # Truncate OCR text if too long (keep first 3000 chars for context)
        ocr_excerpt = ocr_text[:3000] + ("..." if len(ocr_text) > 3000 else "")
        
        prompt = f"""Compare and merge the LLM and NER extraction results for this Korean document.
Output ONLY valid JSON. No markdown, no explanations.

## Original OCR text:
{ocr_excerpt}

## LLM extraction result (structured metadata):
{json.dumps(llm_metadata, ensure_ascii=False, indent=2)}

## NER extraction result (entity list):
{json.dumps(ner_by_type, ensure_ascii=False, indent=2)}

## Field mapping (NER entities mapped to LLM fields):
{json.dumps(field_mappings, ensure_ascii=False, indent=2)}

## Instructions:

1. Compare each field between LLM and NER:
   - Both match → decision: "AGREED"
   - Both exist but differ → decision: "CONFLICT" (check OCR text, pick the more accurate one)
   - Only in LLM → decision: "LLM_ONLY"
   - Only in NER → decision: "NER_ONLY"
   - Neither has it → decision: "MISSING"

2. Select final value:
   - AGREED: use the matched value
   - CONFLICT: reference OCR text to pick the correct one
   - LLM_ONLY: use LLM value (lower confidence)
   - NER_ONLY: use NER value (lower confidence)
   - MISSING: use null

3. Provide reasoning in Korean for each decision.

4. Confidence scores:
   - AGREED: 0.9-1.0
   - CONFLICT resolved: 0.7-0.9
   - LLM_ONLY: 0.5-0.7
   - NER_ONLY: 0.6-0.8
   - MISSING: 0.0

5. Include ALL fields in consolidated_metadata (including null values).

## Required JSON output format:

{{
  "consolidated_metadata": {{
    "field1": "final_value1",
    "field2": "final_value2"
  }},
  "decisions": [
    {{
      "field": "field_name",
      "llm_value": "value_from_llm",
      "ner_value": "value_from_ner",
      "final_value": "selected_value",
      "decision": "AGREED",
      "reasoning": "Korean reasoning here",
      "confidence": 1.0
    }}
  ],
  "summary": {{
    "total_fields": 15,
    "agreed_fields": 12,
    "conflicted_fields": 2,
    "llm_only_fields": 1,
    "ner_only_fields": 0,
    "missing_fields": 0,
    "overall_confidence": 0.92
  }}
}}

JSON response:
"""
        return prompt
    
    def _post_process(
        self,
        consolidated_result: Dict[str, Any],
        llm_result: Dict[str, Any],
        ner_result: Dict[str, Any],
        field_mappings: Dict[str, List],
        validation_errors: List[str]
    ) -> Dict[str, Any]:
        """Post-process consolidation result"""
        
        decisions = consolidated_result.get('decisions', [])
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
                ocr_text=llm_result.get('ocr_text', ''),
                confidence=decision.get('confidence', 0.0)
            )
            
            enhanced_decision['evidence'] = evidence.get('evidence', {})
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
        
        return {
            "success": True,
            "consolidated_metadata": consolidated_result.get('consolidated_metadata', {}),
            "validation_report": validation_report,
            "llm_metadata": llm_result.get('metadata', {}),
            "ner_entities": ner_result.get('extracted_entities', []),
            "model_used": self.model_name,
            "status": consolidated_result.get('status', 'completed'),
            "llm_confidence": consolidated_result.get('llm_confidence', 0.0)
        }
    
    def _fallback_consolidate(
        self,
        llm_metadata: Dict[str, Any],
        ner_entities: List[tuple],
        field_mappings: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Fallback consolidation when LLM call fails
        
        Performs basic merging without LLM assistance
        """
        logger.info("Using fallback consolidation (no LLM)")
        
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
                "overall_confidence": 0.6  # Lower confidence for fallback
            },
            "status": "fallback"
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

