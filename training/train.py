from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.gru_recommender import GRURecommenderConfig, GRUNextCategoryModel
from utils.checkpoint import save_bundle
from utils.data_io import load_activity_log
from utils.dataset import ActivitySequenceDataset, ModelVocab, collate_batch
from utils.features import fit_scaler
from utils.seed import seed_everything
from utils.vocab import fixed_vocab, build_vocab


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 16
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 42


def _time_split_indices(n_rows: int, val_fraction: float) -> tuple[list[int], list[int]]:
    if n_rows < 3:
        raise ValueError("Need at least 3 rows to do train/val split.")
    val_fraction = float(val_fraction)
    if not (0.05 <= val_fraction <= 0.5):
        raise ValueError("val_fraction must be between 0.05 and 0.5")

    split_row = int(round(n_rows * (1.0 - val_fraction)))
    split_row = max(2, min(n_rows - 1, split_row))

    # dataset index i predicts row i+1
    train_idx = list(range(0, max(1, split_row - 1)))
    val_idx = list(range(max(1, split_row - 1), n_rows - 1))
    return train_idx, val_idx


@torch.no_grad()
def _accuracy_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return float((pred == target).float().mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRU next-task-category recommender (offline).")
    parser.add_argument("--data", type=str, required=True, help="Path to activity_log.csv")
    parser.add_argument("--out", type=str, required=True, help="Output model path (.pt)")
    parser.add_argument("--seq-len", type=int, default=TrainConfig.seq_len)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    seed_everything(int(args.seed))

    log = load_activity_log(args.data)
    df = log.df

    category_vocab = build_vocab(df["task_category"].astype(str).tolist(), min_freq=1)
    outcome_vocab = fixed_vocab(["completed", "interrupted"])
    vocab = ModelVocab(category=category_vocab, outcome=outcome_vocab)

    train_idx, val_idx = _time_split_indices(len(df), args.val_fraction)

    # fit scaler on train rows only (no leakage)
    train_rows = df.iloc[: max(train_idx) + 2]  # +2 because index predicts i+1
    duration = train_rows["duration_min"].to_numpy(dtype=np.float32)
    energy = train_rows["energy_level"].to_numpy(dtype=np.float32)
    timestamps = train_rows["timestamp_dt"].tolist()
    delta_min = np.zeros_like(duration, dtype=np.float32)
    for i in range(1, len(timestamps)):
        delta_min[i] = float(max(0.0, (timestamps[i] - timestamps[i - 1]).total_seconds() / 60.0))

    scaler = fit_scaler(duration_min=duration, energy_level=energy, delta_min=delta_min)

    dataset = ActivitySequenceDataset(df=df, vocab=vocab, scaler=scaler, seq_len=int(args.seq_len))
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_batch, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_batch)

    config = GRURecommenderConfig(
        category_vocab_size=len(category_vocab.id_to_token),
        outcome_vocab_size=len(outcome_vocab.id_to_token),
    )
    model = GRUNextCategoryModel(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_path = Path(args.out)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        train_accs: list[float] = []

        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} [train]", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                cat_seq=batch["cat_seq"],
                outcome_seq=batch["outcome_seq"],
                dow_seq=batch["dow_seq"],
                cont_seq=batch["cont_seq"],
                length=batch["length"],
            )
            loss = loss_fn(logits, batch["next_cat"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_accs.append(_accuracy_top1(logits.detach(), batch["next_cat"]))

        model.eval()
        val_losses: list[float] = []
        val_accs: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} [val]", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    cat_seq=batch["cat_seq"],
                    outcome_seq=batch["outcome_seq"],
                    dow_seq=batch["dow_seq"],
                    cont_seq=batch["cont_seq"],
                    length=batch["length"],
                )
                loss = loss_fn(logits, batch["next_cat"])
                val_losses.append(float(loss.item()))
                val_accs.append(_accuracy_top1(logits, batch["next_cat"]))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        mean_train_acc = float(np.mean(train_accs)) if train_accs else float("nan")
        mean_val_acc = float(np.mean(val_accs)) if val_accs else float("nan")

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {mean_train_loss:.4f}, acc {mean_train_acc:.3f} | "
            f"val loss {mean_val_loss:.4f}, acc {mean_val_acc:.3f}"
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            save_bundle(
                best_path,
                model=model.cpu(),
                config=config,
                vocab=vocab,
                scaler=scaler,
                extra={
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "seq_len": int(args.seq_len),
                },
            )
            model.to(device)
            print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    main()

