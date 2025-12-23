from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RecommendationLogRow:
    timestamp: datetime
    energy_level: int | None
    available_minutes: float | None
    topk_categories: list[tuple[str, float]]
    model_path: str | None


def append_recommendation_log(path: str | Path, row: RecommendationLogRow) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = ";".join([f"{c}:{p:.4f}" for c, p in row.topk_categories])

    out_row = {
        "timestamp": row.timestamp.isoformat(sep=" "),
        "energy_level": "" if row.energy_level is None else int(row.energy_level),
        "available_minutes": "" if row.available_minutes is None else float(row.available_minutes),
        "topk_categories": pairs,
        "model_path": "" if row.model_path is None else str(row.model_path),
    }

    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([out_row])], ignore_index=True)
    else:
        df = pd.DataFrame([out_row])
    df.to_csv(path, index=False)

