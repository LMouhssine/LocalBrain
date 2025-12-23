from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class GRURecommenderConfig:
    category_vocab_size: int
    outcome_vocab_size: int
    dow_vocab_size: int = 8  # 0=pad, 1..7=Mon..Sun
    cont_dim: int = 7
    category_emb_dim: int = 32
    outcome_emb_dim: int = 8
    dow_emb_dim: int = 4
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GRURecommenderConfig":
        return GRURecommenderConfig(**d)


class GRUNextCategoryModel(nn.Module):
    def __init__(self, config: GRURecommenderConfig) -> None:
        super().__init__()
        self.config = config

        self.category_emb = nn.Embedding(
            num_embeddings=config.category_vocab_size,
            embedding_dim=config.category_emb_dim,
            padding_idx=0,
        )
        self.outcome_emb = nn.Embedding(
            num_embeddings=config.outcome_vocab_size,
            embedding_dim=config.outcome_emb_dim,
            padding_idx=0,
        )
        self.dow_emb = nn.Embedding(
            num_embeddings=config.dow_vocab_size,
            embedding_dim=config.dow_emb_dim,
            padding_idx=0,
        )

        input_size = config.category_emb_dim + config.outcome_emb_dim + config.dow_emb_dim + config.cont_dim
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=(config.dropout if config.num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.category_vocab_size)

    def forward(
        self,
        *,
        cat_seq: torch.Tensor,
        outcome_seq: torch.Tensor,
        dow_seq: torch.Tensor,
        cont_seq: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        cat_e = self.category_emb(cat_seq)
        out_e = self.outcome_emb(outcome_seq)
        dow_e = self.dow_emb(dow_seq)

        x = torch.cat([cat_e, out_e, dow_e, cont_seq], dim=-1)
        packed = pack_padded_sequence(
            x,
            lengths=length.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        h_last = h_n[-1]
        logits = self.head(self.dropout(h_last))
        return logits

    @torch.inference_mode()
    def predict_proba(self, **kwargs: Any) -> torch.Tensor:
        logits = self.forward(**kwargs)
        return torch.softmax(logits, dim=-1)

