"""
어댑터 기반 NER 모델 (BERT + adapters)
- 저장된 어댑터(models/ner/adapters/*) 자동 로드
- predict(tokens: List[List[str]]) -> BIO label list 반환
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

try:
    from adapters import AutoAdapterModel
except ImportError as e:
    raise RuntimeError("adapters를 설치하세요: pip install -U adapters") from e

from ..config import BIO_LABELS, ID_TO_LABEL


class AdapterNER:
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        model_path: Optional[str] = None,
        adapter_dir: Optional[str] = None,
        active_adapters: Optional[Sequence[str]] = None,  # None이면 디스크에서 전부 로드 후 모두 활성화
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.active_adapters = list(active_adapters) if active_adapters else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        self._load_model()

    def _resolve_model_path(self) -> str:
        # model_path가 있으면 우선
        if self.model_path:
            p = Path(self.model_path)
            return str(p) if p.exists() else self.model_name

        # 없으면 model_downloaded/{model_name} 우선, 없으면 HF
        local = Path("model_downloaded") / self.model_name
        return str(local) if local.exists() else self.model_name

    def _load_model(self) -> None:
        model_path_to_load = self._resolve_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_to_load)
        self.model = AutoAdapterModel.from_pretrained(model_path_to_load)

        # token classification head(이름 고정: "ner")
        self.model.add_tagging_head(
            "ner",
            num_labels=len(BIO_LABELS),
            id2label={i: label for i, label in enumerate(BIO_LABELS)},
        )

        # 토큰 분류 head를 기본 활성 head로 설정
        if hasattr(self.model, "set_active_heads"):
            try:
                self.model.set_active_heads(["ner"])
            except Exception:
                pass
        try:
            self.model.active_head = "ner"
        except Exception:
            if hasattr(self.model, "_active_heads"):
                self.model._active_heads = ["ner"]

        # 저장된 어댑터 로드
        if self.adapter_dir and self.adapter_dir.exists():
            self._load_adapters_from_disk()

        self.model.to(self.device)
        self.model.eval()

    def _load_adapters_from_disk(self) -> None:
        assert self.model is not None
        assert self.adapter_dir is not None

        adapter_names: List[str] = []
        for adapter_path in sorted(self.adapter_dir.iterdir()):
            if not adapter_path.is_dir():
                continue
            # adapters 라이브러리는 adapter_config.json 존재하는 폴더를 보통 어댑터로 봄
            if not (adapter_path / "adapter_config.json").exists():
                continue

            adapter_name = adapter_path.name
            try:
                self.model.load_adapter(str(adapter_path), adapter_name)
                adapter_names.append(adapter_name)
            except Exception:
                # 개별 어댑터 로드 실패는 스킵
                continue

        # 활성화 정책
        if not adapter_names:
            return

        if self.active_adapters:
            # 요청한 것만 활성화
            to_activate = [a for a in self.active_adapters if a in adapter_names]
        else:
            # 전부 활성화(스택)
            to_activate = adapter_names

        if to_activate:
            try:
                self.model.set_active_adapters(to_activate)
            except Exception:
                # set_active_adapters가 실패해도 기본 모델로 추론은 가능
                pass

    @torch.no_grad()
    def predict(self, texts: List[List[str]]) -> List[List[str]]:
        """
        texts: [["기관명",":","주)","나라지식정보"], ...]
        returns: [["O","O","B-company_name","I-company_name"], ...]
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        outputs_all: List[List[str]] = []

        for tokens in texts:
            enc = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = self.model(**enc, head="ner")
            logits = out.logits  # [1, seq, num_labels]
            pred_ids = logits.argmax(dim=-1).cpu().numpy()[0].tolist()

            # word-level alignment: 첫 subword의 라벨만 사용
            word_ids = self.tokenizer(tokens, is_split_into_words=True).word_ids()
            aligned: List[str] = []
            prev = None
            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev:
                    aligned.append(ID_TO_LABEL.get(int(pred_ids[idx]), "O"))
                prev = wid

            # 길이 안전장치
            while len(aligned) < len(tokens):
                aligned.append("O")
            outputs_all.append(aligned[: len(tokens)])

        return outputs_all
