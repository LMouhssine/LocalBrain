from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class TaskSuggestion:
    task_name: str
    score: float
    completion_rate: float
    avg_duration_min: float
    last_seen: datetime | None


def _completion_rate(outcomes: pd.Series) -> float:
    if len(outcomes) == 0:
        return 0.0
    return float((outcomes.astype(str) == "completed").mean())


def suggest_tasks_for_category(
    *,
    df: pd.DataFrame,
    category: str,
    current_energy: int | None,
    available_minutes: float | None,
    top_n: int = 3,
    avoid_last_n: int = 2,
) -> list[TaskSuggestion]:
    d = df[df["task_category"].astype(str) == str(category)].copy()
    if len(d) == 0:
        return []

    recent_names = df["task_name"].astype(str).tail(avoid_last_n).tolist()

    suggestions: list[TaskSuggestion] = []
    for task_name, g in d.groupby(d["task_name"].astype(str)):
        if task_name in recent_names:
            continue
        n = len(g)
        if n < 1:
            continue

        cr = _completion_rate(g["outcome"])
        avg_dur = float(pd.to_numeric(g["duration_min"], errors="coerce").dropna().astype(float).mean())
        last_seen = g["timestamp_dt"].max()

        # Scoring: practical + simple
        score = 0.0
        score += 0.55 * cr

        # Prefer tasks you haven't done very recently
        if isinstance(last_seen, datetime):
            minutes_ago = max(0.0, (df["timestamp_dt"].max() - last_seen).total_seconds() / 60.0)
            recency_score = min(1.0, minutes_ago / (6 * 60))  # saturate at ~6 hours
        else:
            recency_score = 0.5
        score += 0.25 * recency_score

        # Match to available time (if provided)
        if available_minutes is not None and available_minutes > 0:
            diff = abs(avg_dur - float(available_minutes))
            duration_score = max(0.0, 1.0 - (diff / max(float(available_minutes), 30.0)))
        else:
            duration_score = 0.5
        score += 0.20 * duration_score

        # If energy is low, slightly penalize long tasks
        if current_energy is not None and int(current_energy) <= 2:
            if avg_dur > 45:
                score -= 0.10
        elif current_energy is not None and int(current_energy) >= 4:
            if avg_dur < 20:
                score -= 0.05

        suggestions.append(
            TaskSuggestion(
                task_name=task_name,
                score=float(score),
                completion_rate=float(cr),
                avg_duration_min=float(avg_dur),
                last_seen=last_seen if isinstance(last_seen, datetime) else None,
            )
        )

    suggestions.sort(key=lambda s: (-s.score, -s.completion_rate, s.task_name))
    return suggestions[:top_n]

