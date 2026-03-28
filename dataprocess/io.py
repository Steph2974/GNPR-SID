from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


EXPECTED_COLUMNS = [
    "User ID",
    "Venue ID",
    "Venue Category ID",
    "Venue Category Name",
    "Latitude",
    "Longitude",
    "Timezone Offset",
    "UTC Time",
]


def read_raw_dataset_csv(path: str | Path) -> pd.DataFrame:
    """
    Read the original `datasets/{MODE}.csv` with best-effort delimiter/encoding,
    matching the notebook's logic.
    """
    path = Path(path)

    try:
        df = pd.read_csv(path, encoding="utf-8")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep="\t", encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t", encoding="latin-1")
        except Exception:
            df = pd.read_csv(path, encoding="latin-1")

    if len(df.columns) != len(EXPECTED_COLUMNS):
        df.columns = EXPECTED_COLUMNS[: len(df.columns)]
    else:
        df.columns = EXPECTED_COLUMNS

    return df


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_mapping_csv(items: Iterable[tuple[object, int]], path: str | Path, col1: str, col2: str) -> None:
    df = pd.DataFrame(list(items), columns=[col1, col2])
    write_csv(df, path)

