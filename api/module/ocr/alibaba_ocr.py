#!/usr/bin/env python3
"""
Alibaba Cloud OCR Provider
"""

import os
import base64
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional, Dict, Generator
from openai import OpenAI
import module.env_loader  # noqa: F401 — loads .env on import

# Configure logging first
logger = logging.getLogger(__name__)

class AlibabaCloudOCRProvider:
    """Alibaba Cloud DashScope OCR provider."""
    
    # 2026-10-11 재시험(docs/v4/ocr_model_retest_20261011.md — 9월과 같은 40장·앵커·3회):
    #   qwen3.7-plus(생각 끔) 91.7% · 함정 36 · 12.6초  ← 주 모델
    #   qwen3.8-max-0902 89.3% (9월) · qwen3.7-flash(생각 끔) 87.9% · qwen3-vl-235b(퇴역) 87.4% · 함정 87
    #   qwen3.8-flash 86.5% · qwen3-vl-plus 86.5%
    # qwen3-vl-235b-a22b-instruct 는 2026-10-10 Alibaba 공지대로 퇴역했다(계정에서 404 "Model not exist").
    DEFAULT_MODEL = "qwen3.7-plus"
    FALLBACK_MODEL = "qwen3.8-max"

    AVAILABLE_MODELS = {
        "qwen3.7-plus": "Qwen3.7-Plus (권장 · 생각 끔)",
        "qwen3.8-max": "Qwen3.8-Max (폴백 · 생각 끔)",
        "qwen3.8-flash": "Qwen3.8-Flash",
        "qwen3.7-flash": "Qwen3.7-Flash (생각 끔)",
        "qwen3-vl-plus": "Qwen3-VL-Plus",
        "qwen-vl-ocr": "Qwen-VL-OCR (Dedicated OCR)",
        "qwen3.5-plus": "Qwen3.5-Plus (397B, natively multimodal)",
        "qwen3.5-flash": "Qwen3.5-Flash (35B, natively multimodal)",
    }

    # 퇴역 모델(Alibaba 공지 2026-07-08, 2026-10-10 서비스 종료) — 옛 이름이 들어오면(웹 UI 캐시, 외부 클라이언트의
    # ocr_model 값 등) 실패시키지 않고 주 모델로 바꾸고 경고한다.
    RETIRED_MODELS = {
        "qwen3-vl-235b-a22b-instruct": DEFAULT_MODEL, "qwen3-vl-235b-a22b-thinking": DEFAULT_MODEL,
        "qwen3-vl-30b-a3b-instruct": DEFAULT_MODEL, "qwen3-vl-30b-a3b-thinking": DEFAULT_MODEL,
        "qwen3-vl-32b-instruct": DEFAULT_MODEL, "qwen3-vl-8b-instruct": DEFAULT_MODEL,
        "qwen3-vl-flash": DEFAULT_MODEL, "qwen-vl-plus": DEFAULT_MODEL, "qwen-vl-max": DEFAULT_MODEL,
    }

    # 생각(thinking) 모드를 끄고 부르는 모델. qwen3.7 계열은 기본이 생각 켬인데, OCR 에서는 켜면
    # 2배 느리고 출력 토큰이 3배이며 정확도도 낮다(90.0% vs 91.7%). max_tokens 2048 에서는 생각 토큰이 잘림 위험도 만든다.
    THINKING_OFF_MODELS = {"qwen3.7-plus", "qwen3.7-flash", "qwen3.8-max", "qwen3.8-max-0902", "qwen3.6-plus"}

    # 비전 불가 — OCR 로 지정하면 조용히 실패한다(실측 확인, 2026-08/09)
    TEXT_ONLY_MODELS = {
        "qwen3.8-2.4t-a95b",   # HTTP 200 + "이미지를 확인할 수 없어 전사할 수 없습니다" (25자)
        "qwen3.7-max", "qwen3.7-max-preview",
        "qwen3.6-max-preview",  # 이미지 입력 시 HTTP 400
    }

    # 모델 단위 폴백 (1단계만 — 순환 방지)
    OCR_MODEL_FALLBACK = {
        DEFAULT_MODEL: FALLBACK_MODEL,
        FALLBACK_MODEL: DEFAULT_MODEL,
        "qwen3.8-flash": DEFAULT_MODEL,
        "qwen3.7-flash": DEFAULT_MODEL,
        "qwen3-vl-plus": DEFAULT_MODEL,
        "qwen-vl-ocr": DEFAULT_MODEL,
        "qwen3.5-plus": DEFAULT_MODEL,
    }

    # DashScope 의 429 는 **속도 제한**이지 과금 문제가 아니다(본문 문구가 quota 라 오해하기 쉽다).
    # 실측(2026-09-09, qwen3-vl-235b): 동시 4·8 요청은 100% 성공, 동시 12 는 22% 성공.
    # 즉 잠시 기다리면 회복되는 오류다. 예전에는 이걸 곧바로 실패로 처리해
    # 모델 폴백(qwen3.8-flash) → 공급자 폴백(Mistral) 로 밀려났고, 그 세트의 남은 파일까지
    # Mistral 로 처리됐다. 폴백으로 넘기기 전에 같은 모델로 먼저 되쏜다.
    RATE_LIMIT_MAX_RETRIES = 4
    RATE_LIMIT_BASE_DELAY_SEC = 2.0

    def __init__(self, api_key: str, model: Optional[str] = None, region: str = "singapore",
                 temperature: float = 1.0, top_p: float = 0.8, top_k: int = None):
        self.api_key = api_key
        model = model or self.DEFAULT_MODEL
        if model in self.RETIRED_MODELS:
            logger.warning(f"OCR 모델 '{model}' 은(는) 퇴역했습니다(Alibaba, 2026-10-10 서비스 종료) — "
                           f"'{self.RETIRED_MODELS[model]}' 로 바꿔 처리합니다.")
            model = self.RETIRED_MODELS[model]
        self.model = model
        self.region = region
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # 목록은 참고용이며 게이트가 아니다(cloud_extractor.py 와 동일 방침).
        # 다만 **비전 불가 모델만은 막는다.** 실측된 실패 사례가 있기 때문이다:
        # qwen3.8-2.4t-a95b·qwen3.7-max 는 이미지 입력을 오류 없이 받고 HTTP 200 으로
        # "이미지를 확인할 수 없습니다" 라는 유창한 한국어를 돌려준다. OCR 모델로 지정하면
        # 모든 문서의 메타데이터가 조용히 비는데, 어디에서도 오류가 나지 않는다.
        if model in self.TEXT_ONLY_MODELS:
            raise ValueError(
                f"'{model}' 은(는) 텍스트 전용 모델입니다. 이미지 입력을 오류 없이 받지만 "
                f"실제로는 읽지 못하고 정상 응답처럼 보이는 실패를 냅니다. OCR 모델로 쓸 수 없습니다."
            )
        if model not in self.AVAILABLE_MODELS:
            logger.warning(f"OCR 모델 '{model}' 은(는) 알려진 목록에 없습니다. 그대로 사용합니다. "
                           f"비전 지원 여부를 먼저 확인하세요(합성 이미지 1장으로 판독 확인).")
        
        # Map model names to DashScope model IDs
        self.model_mapping = {"qwen-vl-ocr": "qwen-vl-ocr"}   # 그 밖의 이름은 DashScope 모델 ID 그대로
        
        self.dashscope_model_id = self.model_mapping.get(model, model)
        
        # Initialize OpenAI client for DashScope compatible API
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                timeout=60.0,       # 60s timeout per request
                max_retries=3       # retry up to 3 times on transient errors
            )
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
    
    @staticmethod
    def _is_rate_limit(e: Exception) -> bool:
        """429(속도 제한) 여부. 본문에 quota 라고 적혀 있어도 과금 문제가 아니다."""
        return 'RateLimitError' in type(e).__name__ or '429' in str(e)

    @classmethod
    def thinking_params(cls, model_id: str) -> Dict:
        """생각 모드를 꺼야 하는 모델이면 extra_body 를 돌려준다 (그 밖에는 빈 dict — 요청을 바꾸지 않는다)."""
        return {"extra_body": {"enable_thinking": False}} if model_id in cls.THINKING_OFF_MODELS else {}

    def _create_with_retry(self, model_id: str, messages: list, generation_params: Dict):
        """429 만 지수 백오프로 재시도한다. 그 밖의 오류는 그대로 올려 폴백에 맡긴다."""
        generation_params = {**generation_params, **self.thinking_params(model_id)}
        for attempt in range(self.RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return self.client.chat.completions.create(
                    model=model_id, messages=messages, **generation_params)
            except Exception as e:
                if not self._is_rate_limit(e) or attempt == self.RATE_LIMIT_MAX_RETRIES:
                    raise
                delay = self.RATE_LIMIT_BASE_DELAY_SEC * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"DashScope 429 (속도 제한) — {model_id}, {delay:.1f}초 후 재시도 "
                    f"({attempt + 1}/{self.RATE_LIMIT_MAX_RETRIES})")
                time.sleep(delay)

    def process_image(self, image_path: str) -> Dict:
        """Process an image using Alibaba Cloud DashScope API, with model-level fallback."""
        result = self._process_image_with_model(image_path, self.dashscope_model_id)

        # If primary model failed and a fallback exists, try it
        if result.get('metadata', {}).get('error') and self.model in self.OCR_MODEL_FALLBACK:
            fallback_model = self.OCR_MODEL_FALLBACK[self.model]
            primary_error = result['metadata']['error']
            logger.warning(f"OCR model '{self.model}' failed: {primary_error}. Trying fallback: {fallback_model}")
            result = self._process_image_with_model(image_path, fallback_model)
            if not result.get('metadata', {}).get('error'):
                result['metadata']['fallback_used'] = True
                result['metadata']['original_model'] = self.model
                logger.info(f"OCR fallback to '{fallback_model}' succeeded")

        return result

    def _process_image_with_model(self, image_path: str, model_id: str) -> Dict:
        """Process an image with a specific model."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Determine image format
            image_format = image_path.split('.')[-1].lower()
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format == 'png':
                image_format = 'png'
            else:
                image_format = 'jpeg'  # Default fallback
            
            # Prepare messages for OpenAI-compatible API
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}
                    ]
                }
            ]
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 2048
            }
            
            # Note: top_k is not supported by the OpenAI-compatible DashScope endpoint
            
            # Make API call using OpenAI-compatible client (429 는 재시도)
            response = self._create_with_retry(model_id, messages, generation_params)
            
            extracted_text = response.choices[0].message.content
            
            # Clean markdown formatting
            extracted_text = self._clean_markdown_formatting(extracted_text)
            
            return {
                'extracted_text': extracted_text,
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'model': self.dashscope_model_id,
                    'confidence': 0.8,
                    'processing_time': None
                }
            }
            
        except Exception as e:
            # Extract detailed error information
            error_str = str(e)
            error_details = error_str
            rate_limited = False
            
            # Check for common Alibaba Cloud error codes and provide user-friendly messages
            if 'Arrearage' in error_str or 'arrearage' in error_str.lower():
                error_details = "Alibaba Cloud account billing issue: Access denied due to outstanding payment. Please check your account billing status in the Alibaba Cloud console."
            elif 'InvalidApiKey' in error_str or 'invalid' in error_str.lower() and 'key' in error_str.lower():
                error_details = "Invalid Alibaba Cloud API key. Please verify your DASHSCOPE_API_KEY or ALIBABA_API_KEY environment variable."
            elif self._is_rate_limit(e):
                # ⚠️ DashScope 의 429 본문은 "You exceeded your current quota, please check
                # your plan and billing details" 라고 나오지만 **과금 문제가 아니라 속도 제한**이다.
                # 이전 버전은 이 문구의 'quota' 만 보고 "quota exceeded"로 뭉뚱그려
                # 원문을 버렸고, 그 때문에 과금 문제로 오진했다. 원문을 반드시 남긴다.
                # 여기까지 왔다는 건 재시도 4회(약 30초)도 실패했다는 뜻이다.
                rate_limited = True
                error_details = (f"DashScope rate limit (HTTP 429) — 재시도 "
                                 f"{self.RATE_LIMIT_MAX_RETRIES}회 후에도 실패. 동시 요청 수를 줄이세요 "
                                 f"(qwen3-vl-235b 실측 안전선: 동시 8). 과금 문제가 아닙니다. "
                                 f"원문: {error_str[:300]}")
            elif 'QuotaExceeded' in error_str or 'quota' in error_str.lower():
                error_details = f"Alibaba Cloud API quota issue. 원문: {error_str[:300]}"
            elif 'code' in error_str.lower() and ('400' in error_str or '401' in error_str or '403' in error_str):
                # Try to extract error details from string representation
                try:
                    import ast
                    # Look for dictionary in error string
                    if "'error':" in error_str or '"error":' in error_str:
                        # Extract error message from string
                        if "'message':" in error_str:
                            # Find the message part
                            msg_start = error_str.find("'message':") + len("'message':")
                            msg_end = error_str.find(",", msg_start)
                            if msg_end == -1:
                                msg_end = error_str.find("}", msg_start)
                            if msg_end > msg_start:
                                msg = error_str[msg_start:msg_end].strip().strip("'\"")
                                if msg:
                                    error_details = f"Alibaba Cloud API Error: {msg}"
                except:
                    pass
            
            logger.error(f"Alibaba Cloud OCR processing error: {error_details}")
            metadata = {
                'provider': 'alibaba_cloud',
                'model': self.dashscope_model_id,
                'error': error_details,
                'confidence': 0.0
            }
            if rate_limited:
                # 실패 원인이 속도 제한인지 진짜 오류인지 리포트에서 구분할 수 있게 남긴다.
                metadata['rate_limited'] = True
            return {
                'extracted_text': '',
                'metadata': metadata
            }
    
    def process_image_streaming(self, image_path: str) -> Generator[str, None, None]:
        """Process an image with streaming output using OpenAI compatible interface."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Determine image format
            image_format = image_path.split('.')[-1].lower()
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format == 'png':
                image_format = 'png'
            else:
                image_format = 'jpeg'
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}
                    ]
                }
            ]
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 2048,
                "stream": True
            }
            
            # Note: top_k is not supported by the OpenAI-compatible DashScope endpoint
            
            # Make streaming API call
            completion = self.client.chat.completions.create(
                model=self.dashscope_model_id,
                messages=messages,
                **generation_params, **self.thinking_params(self.dashscope_model_id)
            )
            
            full_content = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield content  # Stream output
            
            # Post-process the complete content to remove markdown formatting
            if full_content:
                cleaned_content = self._clean_markdown_formatting(full_content)
                logger.info(f"Streaming completed. Cleaned content length: {len(cleaned_content)}")
            
            logger.info(f"Streaming Alibaba Cloud Qwen-OCR processed {image_path} - {len(full_content)} characters")
            
        except Exception as e:
            logger.error(f"Alibaba Cloud streaming OCR error: {e}")
            yield f"Error: {str(e)}"
    
    def process_image_api_client(self, image_path: str) -> Dict:
        """Process image using API Client approach (non-streaming) with OpenAI compatible interface."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Determine image format
            image_format = image_path.split('.')[-1].lower()
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format == 'png':
                image_format = 'png'
            else:
                image_format = 'jpeg'
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}
                    ]
                }
            ]
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 2048
            }
            
            # Note: top_k is not supported by the OpenAI-compatible DashScope endpoint
            
            # Make API call
            completion = self.client.chat.completions.create(
                model=self.dashscope_model_id,
                messages=messages,
                **generation_params, **self.thinking_params(self.dashscope_model_id)
            )
            
            extracted_text = completion.choices[0].message.content
            
            # Post-process to remove markdown formatting
            extracted_text = self._clean_markdown_formatting(extracted_text)
            
            return {
                'extracted_text': extracted_text,
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'model': self.dashscope_model_id,
                    'confidence': 0.8,
                    'processing_time': None
                }
            }

        except Exception as e:
            logger.error(f"Alibaba Cloud API Client OCR error: {e}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'model': self.dashscope_model_id,
                    'error': str(e),
                    'confidence': 0.0
                }
            }
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting from OCR output."""
        # Remove markdown code blocks
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'```\n?', '', text)
        
        # Remove other markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]*)`', r'\1', text)      # Inline code
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = text.strip()
        
        return text
    
    def get_provider_name(self) -> str:
        """Return the name of the OCR provider."""
        return f"Alibaba Cloud ({self.dashscope_model_id})"
