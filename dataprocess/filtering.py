from __future__ import annotations

import pandas as pd


def filter_low_frequency(df: pd.DataFrame, poi_min_freq: int = 10, user_min_freq: int = 10) -> pd.DataFrame:
    """
    Filter out POIs and users with low frequency.
    Notebook behavior uses `count` (not nunique).
    """
    df = df.copy()

    df["PoiFreq"] = df.groupby("Pid")["Uid"].transform("count")
    df = df[df["PoiFreq"] >= poi_min_freq]

    df["UserFreq"] = df.groupby("Uid")["Pid"].transform("count")
    df = df[df["UserFreq"] >= user_min_freq]

    return df.drop(columns=["PoiFreq", "UserFreq"])

