from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import cos, log1p, pi, sin

import numpy as np


@dataclass(frozen=True)
class FeatureScaler:
    duration_log_mean: float
    duration_log_std: float
    energy_mean: float
    energy_std: float
    delta_log_mean: float
    delta_log_std: float

    def to_dict(self) -> dict[str, float]:
        return {
            "duration_log_mean": self.duration_log_mean,
            "duration_log_std": self.duration_log_std,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "delta_log_mean": self.delta_log_mean,
            "delta_log_std": self.delta_log_std,
        }

    @staticmethod
    def from_dict(d: dict[str, float]) -> "FeatureScaler":
        return FeatureScaler(
            duration_log_mean=float(d["duration_log_mean"]),
            duration_log_std=float(d["duration_log_std"]),
            energy_mean=float(d["energy_mean"]),
            energy_std=float(d["energy_std"]),
            delta_log_mean=float(d["delta_log_mean"]),
            delta_log_std=float(d["delta_log_std"]),
        )


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(values))
    return std if std > 1e-6 else 1.0


def fit_scaler(
    *,
    duration_min: np.ndarray,
    energy_level: np.ndarray,
    delta_min: np.ndarray,
) -> FeatureScaler:
    duration_log = np.log1p(duration_min.astype(np.float32))
    duration_log_mean = float(duration_log.mean())
    duration_log_std = _safe_std(duration_log)

    energy_valid = energy_level[~np.isnan(energy_level)]
    if energy_valid.size == 0:
        energy_mean = 3.0
        energy_std = 1.0
    else:
        energy_mean = float(energy_valid.mean())
        energy_std = _safe_std(energy_valid.astype(np.float32))

    delta_log = np.log1p(delta_min.astype(np.float32))
    delta_log_mean = float(delta_log.mean())
    delta_log_std = _safe_std(delta_log)

    return FeatureScaler(
        duration_log_mean=duration_log_mean,
        duration_log_std=duration_log_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        delta_log_mean=delta_log_mean,
        delta_log_std=delta_log_std,
    )


def time_of_day_sin_cos(ts: datetime) -> tuple[float, float]:
    seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
    frac = seconds / 86400.0
    angle = 2 * pi * frac
    return sin(angle), cos(angle)


def is_weekend(ts: datetime) -> float:
    return 1.0 if ts.weekday() >= 5 else 0.0


def make_cont_features(
    *,
    ts: datetime,
    duration_min: float,
    energy_level: float | None,
    prev_ts: datetime | None,
    scaler: FeatureScaler,
) -> list[float]:
    duration_log = log1p(float(duration_min))
    duration_log_z = (duration_log - scaler.duration_log_mean) / scaler.duration_log_std

    if energy_level is None or np.isnan(energy_level):
        energy_missing = 1.0
        energy = scaler.energy_mean
    else:
        energy_missing = 0.0
        energy = float(energy_level)
    energy_z = (energy - scaler.energy_mean) / scaler.energy_std

    if prev_ts is None:
        delta_min = 0.0
    else:
        delta_min = max(0.0, (ts - prev_ts).total_seconds() / 60.0)
    delta_log = log1p(delta_min)
    delta_log_z = (delta_log - scaler.delta_log_mean) / scaler.delta_log_std

    hour_sin, hour_cos = time_of_day_sin_cos(ts)
    weekend = is_weekend(ts)

    # order matters: keep consistent across training/inference
    return [
        duration_log_z,
        energy_z,
        energy_missing,
        float(hour_sin),
        float(hour_cos),
        float(weekend),
        delta_log_z,
    ]

