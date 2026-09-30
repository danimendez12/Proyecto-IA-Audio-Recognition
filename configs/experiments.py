from __future__ import annotations

from typing import Any


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "balanced",
        "learning_rate": 1e-3,
        "width_mult": 1.0,
        "dropout": 0.2,
        "batch_size": 64,
        "weight_decay": 0.0,
    },
    {
        "name": "regularized",
        "learning_rate": 5e-4,
        "width_mult": 0.75,
        "dropout": 0.35,
        "batch_size": 64,
        "weight_decay": 1e-4,
    },
    {
        "name": "compact",
        "learning_rate": 2e-3,
        "width_mult": 0.5,
        "dropout": 0.1,
        "batch_size": 128,
        "weight_decay": 1e-5,
    },
]