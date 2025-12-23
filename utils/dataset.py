from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.features import FeatureScaler, make_cont_features
from utils.vocab import Vocab


@dataclass(frozen=True)
class ModelVocab:
    category: Vocab
    outcome: Vocab

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": {
                "id_to_token": self.category.id_to_token,
            },
            "outcome": {
                "id_to_token": self.outcome.id_to_token,
            },
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ModelVocab":
        from utils.vocab import Vocab  # avoid circular import

        cat_id_to_token = list(d["category"]["id_to_token"])
        out_id_to_token = list(d["outcome"]["id_to_token"])
        cat_token_to_id = {t: i for i, t in enumerate(cat_id_to_token)}
        out_token_to_id = {t: i for i, t in enumerate(out_id_to_token)}
        return ModelVocab(
            category=Vocab(token_to_id=cat_token_to_id, id_to_token=cat_id_to_token),
            outcome=Vocab(token_to_id=out_token_to_id, id_to_token=out_id_to_token),
        )


class ActivitySequenceDataset(Dataset):
    def __init__(
        self,
        *,
        df,
        vocab: ModelVocab,
        scaler: FeatureScaler,
        seq_len: int,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.scaler = scaler
        self.seq_len = int(seq_len)
        self.n_rows = len(self.df)

        if self.n_rows < 2:
            raise ValueError("Need at least 2 log rows to build (context -> next) examples.")

        categories = self.df["task_category"].astype(str).tolist()
        outcomes = self.df["outcome"].astype(str).tolist()
        timestamps = self.df["timestamp_dt"].tolist()
        durations = self.df["duration_min"].to_numpy(dtype=np.float32)
        energy = self.df["energy_level"].to_numpy(dtype=np.float32)

        self.cat_ids = np.array([self.vocab.category.encode(cat) for cat in categories], dtype=np.int64)
        self.out_ids = np.array([self.vocab.outcome.encode(out) for out in outcomes], dtype=np.int64)
        self.dow_ids = np.array([int(ts.weekday()) + 1 for ts in timestamps], dtype=np.int64)

        cont_rows: list[list[float]] = []
        prev_ts = None
        for ts, duration_min, energy_level in zip(timestamps, durations, energy):
            cont_rows.append(
                make_cont_features(
                    ts=ts,
                    duration_min=float(duration_min),
                    energy_level=(None if np.isnan(energy_level) else float(energy_level)),
                    prev_ts=prev_ts,
                    scaler=self.scaler,
                )
            )
            prev_ts = ts
        self.cont = np.array(cont_rows, dtype=np.float32)

    def __len__(self) -> int:
        # each row (except the first) can be predicted from previous history
        return self.n_rows - 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # target is next category for row idx+1
        end = idx + 1
        start = max(0, end - self.seq_len)
        length = end - start
        pad = self.seq_len - length
        assert pad >= 0

        cat_seq = np.full(self.seq_len, self.vocab.category.pad_id, dtype=np.int64)
        out_seq = np.full(self.seq_len, self.vocab.outcome.pad_id, dtype=np.int64)
        dow_seq = np.zeros(self.seq_len, dtype=np.int64)
        cont_seq = np.zeros((self.seq_len, self.cont.shape[1]), dtype=np.float32)

        cat_seq[pad:] = self.cat_ids[start:end]
        out_seq[pad:] = self.out_ids[start:end]
        dow_seq[pad:] = self.dow_ids[start:end]
        cont_seq[pad:, :] = self.cont[start:end]

        next_cat = int(self.cat_ids[idx + 1])

        return {
            "cat_seq": torch.from_numpy(cat_seq),
            "outcome_seq": torch.from_numpy(out_seq),
            "dow_seq": torch.from_numpy(dow_seq),
            "cont_seq": torch.from_numpy(cont_seq),
            "length": torch.tensor(length, dtype=torch.long),
            "next_cat": torch.tensor(next_cat, dtype=torch.long),
        }


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
