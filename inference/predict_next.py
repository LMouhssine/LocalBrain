from __future__ import annotations

import argparse

import numpy as np
import torch

from utils.checkpoint import load_bundle
from utils.context import build_recent_context_tensors
from utils.data_io import load_activity_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next task category from recent activity (offline).")
    parser.add_argument("--data", type=str, required=True, help="Path to activity_log.csv")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model bundle (.pt)")
    parser.add_argument("--k", type=int, default=5, help="Top-k categories to display")
    parser.add_argument("--seq-len", type=int, default=0, help="Context length override (0 = use trained default)")
    args = parser.parse_args()

    log = load_activity_log(args.data)
    bundle = load_bundle(args.model, map_location="cpu")
    device = torch.device("cpu")
    bundle.model.to(device)

    seq_len = int(args.seq_len) if int(args.seq_len) > 0 else int(bundle.extra.get("seq_len", 16))

    batch = build_recent_context_tensors(
        df=log.df,
        vocab=bundle.vocab,
        scaler=bundle.scaler,
        seq_len=seq_len,
        device=device,
    )

    proba = bundle.model.predict_proba(**batch)[0].detach().cpu().numpy().astype(np.float64)
    proba[bundle.vocab.category.pad_id] = 0.0
    proba[bundle.vocab.category.unk_id] = 0.0

    k = max(1, int(args.k))
    top_ids = np.argsort(-proba)[:k]

    print("Top recommendations (next task category):")
    for rank, cat_id in enumerate(top_ids, start=1):
        cat = bundle.vocab.category.decode(int(cat_id))
        p = float(proba[int(cat_id)])
        print(f"{rank:02d}. {cat}  (p={p:.3f})")


if __name__ == "__main__":
    main()

