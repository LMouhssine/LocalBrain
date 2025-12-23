from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from utils.data_io import append_activity_row, load_activity_log
from utils.insights import describe_insights, format_insights
from utils.next_category import next_categories
from utils.recommendation import suggest_tasks_for_category
from utils.recommendation_log import RecommendationLogRow, append_recommendation_log


def _prompt(text: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{text}{suffix}: ").strip()
    return value if value else (default or "")


def _prompt_int(text: str, *, default: int | None = None) -> int | None:
    raw = _prompt(text, default=("" if default is None else str(default))).strip()
    if raw == "":
        return None
    return int(raw)


def _prompt_float(text: str, *, default: float | None = None) -> float | None:
    raw = _prompt(text, default=("" if default is None else str(default))).strip()
    if raw == "":
        return None
    return float(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily offline recommender (log -> predict -> suggest).")
    parser.add_argument("--data", type=str, required=True, help="Path to activity_log.csv")
    parser.add_argument("--model", type=str, default="", help="Path to saved model bundle (.pt). Optional.")
    parser.add_argument("--k", type=int, default=5, help="Top-k categories to show")
    parser.add_argument("--seq-len", type=int, default=0, help="Context length override (0 = trained default)")
    parser.add_argument("--energy", type=int, default=0, help="Current energy 1-5 (0 = prompt/unknown)")
    parser.add_argument("--available-minutes", type=float, default=0.0, help="Time available now (0 = unknown)")
    parser.add_argument("--log-activity", action="store_true", help="Prompt and append the most recent activity row.")
    parser.add_argument("--log-recommendation", action="store_true", help="Append recommendation to data/recommendation_log.csv")
    parser.add_argument("--no-insights", action="store_true", help="Skip printing productivity insights")
    args = parser.parse_args()

    data_path = Path(args.data)
    now = datetime.now()

    if args.log_activity:
        ts_raw = _prompt("Timestamp (ISO)", default=now.isoformat(sep=" "))
        ts = datetime.fromisoformat(ts_raw)
        task_name = _prompt("Task name")
        task_category = _prompt("Task category")
        duration_min = float(_prompt_float("Duration (minutes)", default=25.0) or 25.0)
        energy_level = _prompt_int("Energy level 1-5 (blank if unknown)", default=None)
        outcome = _prompt("Outcome (completed/interrupted)", default="completed").strip().lower()
        if outcome not in {"completed", "interrupted"}:
            raise ValueError("Outcome must be 'completed' or 'interrupted'")
        if energy_level is not None and energy_level not in {1, 2, 3, 4, 5}:
            raise ValueError("Energy level must be 1-5 (or blank)")

        append_activity_row(
            data_path,
            timestamp=ts,
            task_name=task_name,
            task_category=task_category,
            duration_min=duration_min,
            energy_level=energy_level,
            outcome=outcome,
        )
        print(f"Appended activity to {data_path}")

    log = load_activity_log(data_path)
    df = log.df

    energy = int(args.energy) if int(args.energy) in (1, 2, 3, 4, 5) else None
    if energy is None and args.energy != 0:
        raise ValueError("--energy must be 0 or in {1,2,3,4,5}")
    if energy is None:
        energy = _prompt_int("Current energy 1-5 (blank if unknown)", default=None)

    available_minutes = float(args.available_minutes) if float(args.available_minutes) > 0 else None
    if available_minutes is None:
        available_minutes = _prompt_float("Available minutes now (blank if unknown)", default=None)

    if not args.no_insights:
        insights = describe_insights(df)
        print("\n=== Your patterns (from the log) ===")
        print(format_insights(insights))

    model_path = Path(args.model) if args.model else None

    topk, _source = next_categories(
        df,
        model_path=(str(model_path) if model_path is not None else None),
        k=max(1, int(args.k)),
        seq_len_override=int(args.seq_len),
    )

    if not topk:
        print("Not enough data to recommend yet. Add a few log entries and try again.")
        return

    print("\n=== Recommendation ===")
    print("Top categories:")
    for rank, (cat, p) in enumerate(topk, start=1):
        print(f"{rank:02d}. {cat}  (score={p:.3f})")

    best_category = topk[0][0]
    suggestions = suggest_tasks_for_category(
        df=df,
        category=best_category,
        current_energy=energy,
        available_minutes=available_minutes,
        top_n=3,
    )

    if suggestions:
        print(f"\nSuggested next tasks in '{best_category}':")
        for s in suggestions:
            last_seen = "n/a" if s.last_seen is None else s.last_seen.isoformat(sep=" ")
            print(
                f"- {s.task_name} | score={s.score:.2f} | "
                f"completion={s.completion_rate:.0%} | avg_dur={s.avg_duration_min:.0f}m | last={last_seen}"
            )
    else:
        print(f"\nNo task-name history for category '{best_category}'. Add a few task_name entries to improve this.")

    if args.log_recommendation:
        rec_path = data_path.parent / "recommendation_log.csv"
        append_recommendation_log(
            rec_path,
            RecommendationLogRow(
                timestamp=now,
                energy_level=energy,
                available_minutes=available_minutes,
                topk_categories=topk,
                model_path=(str(model_path) if model_path is not None else None),
            ),
        )
        print(f"\nLogged recommendation to {rec_path}")


if __name__ == "__main__":
    main()
