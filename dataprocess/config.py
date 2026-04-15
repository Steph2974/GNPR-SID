from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str = "NYC"

    # I/O: raw CSV at ``data_root / f"{dataset}.csv"``; all outputs under ``output_root / dataset /``
    data_root: Path = Path("datasets")
    output_root: Path | None = None

    # cell1: filter thresholds
    poi_min_freq: int = 10
    user_min_freq: int = 10

    # cell5: sequence generation
    window_size: int = 50
    step_size: int = 10
    mask_prob: float = 0.1

