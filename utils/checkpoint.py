from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from model.gru_recommender import GRURecommenderConfig, GRUNextCategoryModel
from utils.dataset import ModelVocab
from utils.features import FeatureScaler


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelBundle:
    model: GRUNextCategoryModel
    config: GRURecommenderConfig
    vocab: ModelVocab
    scaler: FeatureScaler
    extra: dict[str, Any]


def save_bundle(
    path: str | Path,
    *,
    model: GRUNextCategoryModel,
    config: GRURecommenderConfig,
    vocab: ModelVocab,
    scaler: FeatureScaler,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "vocab": vocab.to_dict(),
        "scaler": scaler.to_dict(),
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_bundle(path: str | Path, *, map_location: str | torch.device = "cpu") -> ModelBundle:
    path = Path(path)
    payload = torch.load(path, map_location=map_location)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model schema_version={payload.get('schema_version')}. Expected {SCHEMA_VERSION}."
        )

    config = GRURecommenderConfig.from_dict(payload["config"])
    vocab = ModelVocab.from_dict(payload["vocab"])
    scaler = FeatureScaler.from_dict(payload["scaler"])

    model = GRUNextCategoryModel(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    extra = dict(payload.get("extra") or {})
    created_at_utc = payload.get("created_at_utc")
    if isinstance(created_at_utc, str) and created_at_utc:
        extra.setdefault("created_at_utc", created_at_utc)
    extra.setdefault("schema_version", payload.get("schema_version"))
    return ModelBundle(model=model, config=config, vocab=vocab, scaler=scaler, extra=extra)
