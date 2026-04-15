from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from .filtering import filter_low_frequency
from .io import read_raw_dataset_csv, write_csv


def prepare_and_filter_raw_dataset(
    dataset: str,
    poi_min_freq: int,
    user_min_freq: int,
    *,
    raw_csv: Path,
    out_filtered_csv: Path,
) -> Path:
    """
    Read ``raw_csv``, add Region (Plus Code) / Local Time, rename columns,
    filter by frequency, and write ``out_filtered_csv``.

    Returns the output path.
    """
    df = read_raw_dataset_csv(raw_csv)

    # optional dependency, so import lazily
    from .geo import pluscode6

    df["Region"] = df.apply(lambda row: pluscode6(row["Latitude"], row["Longitude"]), axis=1)
    df["UTC Time"] = pd.to_datetime(df["UTC Time"], format="%a %b %d %H:%M:%S %z %Y")
    df["Local Time"] = (df["UTC Time"] + df["Timezone Offset"].apply(lambda x: timedelta(minutes=x))).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    df.columns = ["Uid", "Pid", "Venue Category ID", "Catname", "Lat", "Lon", "Timezone Offset", "UTC Time", "Region", "Time"]
    df = df[["Uid", "Pid", "Catname", "Region", "Lat", "Lon", "Time"]]

    df = filter_low_frequency(df, poi_min_freq=poi_min_freq, user_min_freq=user_min_freq)

    write_csv(df, out_filtered_csv)
    return out_filtered_csv

