from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HourStat:
    hour: int
    n: int
    completion_rate: float
    avg_energy: float | None


@dataclass(frozen=True)
class CategoryStat:
    category: str
    n: int
    completion_rate: float
    avg_duration_min: float
    avg_energy: float | None


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(values) == 0:
        return None
    return float(values.mean())


def hourly_stats(df: pd.DataFrame) -> list[HourStat]:
    hours = df["timestamp_dt"].apply(lambda d: int(d.hour))
    completed = (df["outcome"] == "completed").astype(int)

    out: list[HourStat] = []
    for hour in range(24):
        mask = hours == hour
        n = int(mask.sum())
        if n == 0:
            continue
        completion_rate = float(completed[mask].mean())
        avg_energy = _safe_mean(df.loc[mask, "energy_level"])
        out.append(
            HourStat(
                hour=hour,
                n=n,
                completion_rate=completion_rate,
                avg_energy=avg_energy,
            )
        )
    return out


def category_stats(df: pd.DataFrame) -> list[CategoryStat]:
    out: list[CategoryStat] = []
    for category, g in df.groupby("task_category"):
        n = int(len(g))
        completion_rate = float((g["outcome"] == "completed").mean())
        avg_duration_min = float(pd.to_numeric(g["duration_min"], errors="coerce").dropna().astype(float).mean())
        avg_energy = _safe_mean(g["energy_level"])
        out.append(
            CategoryStat(
                category=str(category),
                n=n,
                completion_rate=completion_rate,
                avg_duration_min=avg_duration_min,
                avg_energy=avg_energy,
            )
        )
    out.sort(key=lambda s: (-s.n, s.category))
    return out


def find_low_productivity_hours(
    hour_stats_list: list[HourStat], *, min_samples: int = 5, top_k: int = 3
) -> list[HourStat]:
    eligible = [h for h in hour_stats_list if h.n >= min_samples]
    eligible.sort(key=lambda h: (h.completion_rate, -h.n))
    return eligible[:top_k]


def describe_insights(df: pd.DataFrame) -> dict[str, object]:
    hstats = hourly_stats(df)
    cstats = category_stats(df)
    low = find_low_productivity_hours(hstats)

    overall_completion = float((df["outcome"] == "completed").mean()) if len(df) else float("nan")
    overall_energy = _safe_mean(df["energy_level"])

    return {
        "overall_completion_rate": overall_completion,
        "overall_avg_energy": overall_energy,
        "hourly": hstats,
        "category": cstats,
        "low_productivity_hours": low,
    }


def format_insights(insights: dict[str, object]) -> str:
    overall_completion = insights.get("overall_completion_rate")
    overall_energy = insights.get("overall_avg_energy")
    low = insights.get("low_productivity_hours", [])

    lines: list[str] = []
    if isinstance(overall_completion, (float, int)) and not np.isnan(overall_completion):
        lines.append(f"Overall completion rate: {overall_completion:.0%}")
    if overall_energy is not None:
        lines.append(f"Overall avg energy: {overall_energy:.2f}")

    if low:
        lines.append("Likely low-productivity hours (based on interruptions):")
        for h in low:
            avg_energy = "n/a" if h.avg_energy is None else f"{h.avg_energy:.2f}"
            lines.append(f"  - {h.hour:02d}:00 (n={h.n}, completion={h.completion_rate:.0%}, avg_energy={avg_energy})")

    return "\n".join(lines)

