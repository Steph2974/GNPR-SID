from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd


def sequences_by_user(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with columns: Uid, Pid(list), Catname(list)."""
    return (
        df.groupby("Uid")
        .agg(
            {
                "Pid": list,
                "Catname": list,
            }
        )
        .reset_index()
    )


def get_forward_neighbors(sequences: Iterable[list[int]], min_freq: int = 1) -> dict[int, list[int]]:
    neighbor_counts = defaultdict(Counter)
    all_pois: set[int] = set()

    for seq in sequences:
        all_pois.update(seq)
        for i in range(len(seq) - 1):
            neighbor_counts[seq[i]][seq[i + 1]] += 1

    out: dict[int, list[int]] = {}
    for poi in all_pois:
        counter = neighbor_counts.get(poi, Counter())
        filtered = {n: f for n, f in counter.items() if f >= min_freq}
        out[poi] = [n for n, _ in sorted(filtered.items(), key=lambda x: x[1], reverse=True)] if filtered else []
    return out


def get_neighbors(sequences: Iterable[list[int]], min_freq: int = 1) -> dict[int, list[int]]:
    neighbor_counts = defaultdict(Counter)
    all_pois: set[int] = set()

    for seq in sequences:
        all_pois.update(seq)
        for i, poi in enumerate(seq):
            if i > 0:
                neighbor_counts[poi][seq[i - 1]] += 1
            if i < len(seq) - 1:
                neighbor_counts[poi][seq[i + 1]] += 1

    out: dict[int, list[int]] = {}
    for poi in all_pois:
        counter = neighbor_counts.get(poi, Counter())
        filtered = {n: f for n, f in counter.items() if f >= min_freq}
        out[poi] = [n for n, _ in sorted(filtered.items(), key=lambda x: x[1], reverse=True)]
    return out

