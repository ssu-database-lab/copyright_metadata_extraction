"""Bi-LSTM + CRF 모델 클래스"""
from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn

from .config import BIO_LABELS


class BiLSTMCRF(nn.Module):
    """Bi-LSTM + CRF 모델"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_labels: Optional[int] = None):
        super().__init__()
        if num_labels is None:
            num_labels = len(BIO_LABELS) if BIO_LABELS else 13
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.crf = CRFLayer(num_labels)
    
    def forward(self, x, labels=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        
        if labels is not None:
            loss = -self.crf(emissions, labels)
            return loss
        else:
            return self.crf.decode(emissions)
    
    def predict(self, texts: List[List[str]]) -> List[List[str]]:
        """예측 (구현 필요)"""
        raise NotImplementedError("BiLSTMCRF 예측은 아직 구현되지 않았습니다.")


class CRFLayer(nn.Module):
    """CRF 레이어"""
    
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """CRF forward (log-likelihood)"""
        batch_size, seq_len = tags.shape
        score = torch.zeros(batch_size)
        
        for i in range(batch_size):
            for j in range(seq_len - 1):
                score[i] += self.transitions[tags[i, j], tags[i, j + 1]]
                score[i] += emissions[i, j, tags[i, j]]
            score[i] += emissions[i, seq_len - 1, tags[i, seq_len - 1]]
        
        return score.sum()
    
    def decode(self, emissions: torch.Tensor) -> List[List[int]]:
        """Viterbi 디코딩 (간단한 greedy 구현)"""
        batch_size, seq_len, num_labels = emissions.shape
        best_paths = []
        
        for i in range(batch_size):
            path = []
            for j in range(seq_len):
                path.append(emissions[i, j].argmax().item())
            best_paths.append(path)
        
        return best_paths

