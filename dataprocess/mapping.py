from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd

from .io import write_csv, write_mapping_csv


@dataclass(frozen=True)
class Mappings:
    uid_map: dict[object, int]
    pid_map: dict[object, int]
    cat_map: dict[object, int]
    reg_map: dict[object, int]


def _make_mapping(values: list[object], seed: int | None = None) -> dict[object, int]:
    values = list(values)
    rng = random.Random(seed)
    rng.shuffle(values)
    return {v: i for i, v in enumerate(values, start=1)}


def map_ids(
    dataset: str,
    input_csv: str | Path,
    out_dir: str | Path,
    *,
    seed: int | None = 0,
    force_region_zero: bool = False,
) -> tuple[pd.DataFrame, Mappings]:
    """
    Read filtered `{dataset}.csv`, shuffle-and-map Uid/Pid/Catname/Region to integers,
    write mapping CSVs, and write `data.csv` into `out_dir`.
    """
    df = pd.read_csv(input_csv)

    uid_map = _make_mapping(list(df["Uid"].unique()), seed=seed)
    pid_map = _make_mapping(list(df["Pid"].unique()), seed=seed + 1 if seed is not None else None)
    cat_map = _make_mapping(list(df["Catname"].unique()), seed=seed + 2 if seed is not None else None)

    if force_region_zero:
        reg_map = {reg: 0 for reg in df["Region"].unique()}
    else:
        reg_map = _make_mapping(list(df["Region"].unique()), seed=seed + 3 if seed is not None else None)

    df["Uid"] = df["Uid"].map(uid_map)
    df["Pid"] = df["Pid"].map(pid_map)
    df["Catname"] = df["Catname"].map(cat_map)
    df["Region"] = df["Region"].map(reg_map)

    out_dir = Path(out_dir)
    write_mapping_csv(uid_map.items(), out_dir / "uid_mapping.csv", "Original_Uid", "Mapped_Uid")
    write_mapping_csv(pid_map.items(), out_dir / "pid_mapping.csv", "Original_Pid", "Mapped_Pid")
    write_mapping_csv(cat_map.items(), out_dir / "catname_mapping.csv", "Original_Catname", "Mapped_Catname")
    write_mapping_csv(reg_map.items(), out_dir / "region_mapping.csv", "Original_Region", "Mapped_Region")

    write_csv(df, out_dir / "data.csv")
    return df, Mappings(uid_map=uid_map, pid_map=pid_map, cat_map=cat_map, reg_map=reg_map)

