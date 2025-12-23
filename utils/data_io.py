from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "task_name",
    "task_category",
    "duration_min",
    "outcome",
]

OPTIONAL_COLUMNS = ["energy_level"]

ALLOWED_OUTCOMES = {"completed", "interrupted"}


@dataclass(frozen=True)
class ActivityLog:
    df: pd.DataFrame


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        raise ValueError("timestamp is missing")
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp is blank")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        # common alternative: "YYYY-MM-DD HH:MM"
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        raise


def load_activity_log(csv_path: str | Path) -> ActivityLog:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Activity log not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. See data/README.md")

    # normalize optional columns
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.copy()
    df["timestamp_dt"] = df["timestamp"].apply(_parse_timestamp)

    df["task_name"] = df["task_name"].astype(str).str.strip()
    df["task_category"] = df["task_category"].astype(str).str.strip()

    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    if df["duration_min"].isna().any():
        bad = df[df["duration_min"].isna()][["timestamp", "task_name", "duration_min"]].head(5)
        raise ValueError(f"Some duration_min values are not numeric. Examples:\n{bad}")
    if (df["duration_min"] <= 0).any():
        bad = df[df["duration_min"] <= 0][["timestamp", "task_name", "duration_min"]].head(5)
        raise ValueError(f"Some duration_min values are <= 0. Examples:\n{bad}")

    df["outcome"] = df["outcome"].astype(str).str.strip().str.lower()
    if not df["outcome"].isin(ALLOWED_OUTCOMES).all():
        bad = sorted(set(df.loc[~df["outcome"].isin(ALLOWED_OUTCOMES), "outcome"].tolist()))
        raise ValueError(f"Invalid outcome values: {bad}. Allowed: {sorted(ALLOWED_OUTCOMES)}")

    # energy: optional, allow blanks
    df["energy_level"] = pd.to_numeric(df["energy_level"], errors="coerce")
    if ((df["energy_level"] < 1) | (df["energy_level"] > 5)).any():
        bad = df[(df["energy_level"] < 1) | (df["energy_level"] > 5)][
            ["timestamp", "task_name", "energy_level"]
        ].head(5)
        raise ValueError("energy_level must be 1-5 (or blank). Examples:\n" + str(bad))

    df = df.sort_values("timestamp_dt").reset_index(drop=True)
    return ActivityLog(df=df)


def append_activity_row(
    csv_path: str | Path,
    *,
    timestamp: datetime,
    task_name: str,
    task_category: str,
    duration_min: float,
    outcome: str,
    energy_level: int | None = None,
) -> None:
    csv_path = Path(csv_path)
    row = {
        "timestamp": timestamp.isoformat(sep=" "),
        "task_name": task_name,
        "task_category": task_category,
        "duration_min": float(duration_min),
        "energy_level": "" if energy_level is None else int(energy_level),
        "outcome": outcome,
    }
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
