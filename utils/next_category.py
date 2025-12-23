from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.checkpoint import ModelBundle, load_bundle
from utils.context import build_recent_context_tensors


def heuristic_next_categories(df: pd.DataFrame, *, k: int) -> list[tuple[str, float]]:
    k = max(1, int(k))
    if len(df) < 2:
        cats = df["task_category"].astype(str).tolist()
        if not cats:
            return []
        return [(cats[-1], 1.0)]

    prev = df["task_category"].astype(str).iloc[:-1].tolist()
    nxt = df["task_category"].astype(str).iloc[1:].tolist()
    last_cat = str(df["task_category"].iloc[-1])

    counts: dict[str, int] = {}
    total = 0
    for p, n in zip(prev, nxt):
        if p != last_cat:
            continue
        counts[n] = counts.get(n, 0) + 1
        total += 1

    if total == 0:
        overall = df["task_category"].astype(str).value_counts().head(k)
        s = float(overall.sum()) if len(overall) else 1.0
        return [(c, float(v) / s) for c, v in overall.items()]

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [(c, float(v) / float(total)) for c, v in ranked]


@torch.inference_mode()
def model_next_categories(
    df: pd.DataFrame, *, bundle: ModelBundle, k: int, seq_len: int
) -> list[tuple[str, float]]:
    device = torch.device("cpu")
    bundle.model.to(device)
    batch = build_recent_context_tensors(
        df=df,
        vocab=bundle.vocab,
        scaler=bundle.scaler,
        seq_len=int(seq_len),
        device=device,
    )
    proba = bundle.model.predict_proba(**batch)[0].detach().cpu().numpy().astype(np.float64)
    proba[bundle.vocab.category.pad_id] = 0.0
    proba[bundle.vocab.category.unk_id] = 0.0

    k = max(1, int(k))
    top_ids = np.argsort(-proba)[:k]
    return [(bundle.vocab.category.decode(int(i)), float(proba[int(i)])) for i in top_ids]


def next_categories(
    df: pd.DataFrame,
    *,
    model_path: str | Path | None,
    bundle: ModelBundle | None = None,
    k: int,
    seq_len_override: int = 0,
    default_seq_len: int = 16,
) -> tuple[list[tuple[str, float]], str]:
    """
    Returns (topk, source) where source is "model" or "heuristic".
    """
    if bundle is None and model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            bundle = load_bundle(model_path, map_location="cpu")

    if bundle is not None:
        seq_len = int(seq_len_override) if int(seq_len_override) > 0 else int(bundle.extra.get("seq_len", default_seq_len))
        return model_next_categories(df, bundle=bundle, k=k, seq_len=seq_len), "model"

    return heuristic_next_categories(df, k=k), "heuristic"
