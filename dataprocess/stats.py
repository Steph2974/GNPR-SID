from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HistoryStats:
    average_history_length: float
    most_frequent_length: int


def history_length_stats(data_csv: str | Path) -> HistoryStats:
    """
    Notebook cell6:
    compute per-user history length stats on `datasets/{dataset}/data.csv`.
    """
    df = pd.read_csv(data_csv, usecols=["Uid", "Pid", "Time"])
    user_history_length = df.groupby("Uid").size()
    avg = float(user_history_length.mean())
    most = int(user_history_length.value_counts().idxmax())
    return HistoryStats(average_history_length=avg, most_frequent_length=most)

