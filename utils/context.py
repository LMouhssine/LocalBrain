from __future__ import annotations

import numpy as np
import torch

from utils.dataset import ModelVocab
from utils.features import FeatureScaler, make_cont_features


def build_recent_context_tensors(
    *,
    df,
    vocab: ModelVocab,
    scaler: FeatureScaler,
    seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(df) == 0:
        raise ValueError("Activity log is empty.")
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    end = len(df)
    start = max(0, end - seq_len)
    window = df.iloc[start:end].reset_index(drop=True)
    pad = seq_len - len(window)
    assert pad >= 0

    cat_ids = [vocab.category.pad_id] * pad
    out_ids = [vocab.outcome.pad_id] * pad
    dow_ids = [0] * pad
    cont = [[0.0] * 7 for _ in range(pad)]

    prev_ts = None
    if len(window) > 0 and start > 0:
        prev_ts = df.iloc[start - 1]["timestamp_dt"]

    for _, row in window.iterrows():
        ts = row["timestamp_dt"]
        cat_ids.append(vocab.category.encode(row["task_category"]))
        out_ids.append(vocab.outcome.encode(row["outcome"]))
        dow_ids.append(int(ts.weekday()) + 1)
        cont.append(
            make_cont_features(
                ts=ts,
                duration_min=float(row["duration_min"]),
                energy_level=(None if np.isnan(row["energy_level"]) else float(row["energy_level"])),
                prev_ts=prev_ts,
                scaler=scaler,
            )
        )
        prev_ts = ts

    length = len(window)
    batch = {
        "cat_seq": torch.tensor([cat_ids], dtype=torch.long, device=device),
        "outcome_seq": torch.tensor([out_ids], dtype=torch.long, device=device),
        "dow_seq": torch.tensor([dow_ids], dtype=torch.long, device=device),
        "cont_seq": torch.tensor([cont], dtype=torch.float32, device=device),
        "length": torch.tensor([length], dtype=torch.long, device=device),
    }
    return batch

