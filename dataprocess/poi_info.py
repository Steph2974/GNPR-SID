from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .io import write_csv
from .neighbors import get_forward_neighbors, get_neighbors, sequences_by_user


def build_poi_info(dataset: str, data_csv: str | Path, out_path: str | Path) -> pd.DataFrame:
    """
    Build `poi_info.csv` compatible with `code/POIdataset.py`.
    Output columns include:
    Pid, Uid(list), Catname, Original_Catname, Region, Lat, Lon, Time(list[int-hour]), neighbors(list), forward_neighbors(list)

    ``Original_Catname`` is the raw category string before integer mapping, resolved via
    ``catname_mapping.csv`` next to ``data_csv`` when present; otherwise it equals ``Catname``.
    """
    df = pd.read_csv(data_csv)
    df["Time"] = pd.to_datetime(df["Time"]).dt.hour

    seq_df = sequences_by_user(df)
    sequences = seq_df["Pid"].tolist()

    poi_info = (
        df.groupby("Pid")
        .agg(
            {
                "Uid": list,
                "Catname": lambda x: x.iloc[0],
                "Region": lambda x: x.iloc[0],
                "Lat": lambda x: x.iloc[0],
                "Lon": lambda x: x.iloc[0],
                "Time": list,
            }
        )
        .reset_index()
    )

    poi_info["Uid"] = poi_info["Uid"].apply(lambda uids: [uid for uid, c in Counter(uids).items() if c >= 1])
    poi_info["Time"] = poi_info["Time"].apply(lambda times: [t for t, c in Counter(times).items() if c >= 1])

    data_dir = Path(data_csv).parent
    mapping_path = data_dir / "catname_mapping.csv"
    if mapping_path.is_file():
        cat_map_df = pd.read_csv(mapping_path)
        id_to_orig = dict(zip(cat_map_df["Mapped_Catname"], cat_map_df["Original_Catname"]))
        original_cat = poi_info["Catname"].map(id_to_orig)
    else:
        original_cat = poi_info["Catname"]
    loc = int(poi_info.columns.get_loc("Catname")) + 1
    poi_info.insert(loc, "Original_Catname", original_cat)

    nb = get_neighbors(sequences, min_freq=1)
    fnb = get_forward_neighbors(sequences, min_freq=1)
    poi_info["neighbors"] = poi_info["Pid"].map(lambda pid: nb.get(pid, []))
    poi_info["forward_neighbors"] = poi_info["Pid"].map(lambda pid: fnb.get(pid, []))

    out_path = Path(out_path)
    write_csv(poi_info, out_path)
    return poi_info

