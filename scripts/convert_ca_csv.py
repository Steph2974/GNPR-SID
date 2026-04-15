#!/usr/bin/env python3
"""
Convert raw CA.csv (UserId, PoiId, PoiCategoryId JSON, Lat, Lon, UTCTime)
into the 8-column Foursquare-style CSV expected by dataprocess/io.read_raw_dataset_csv.

PoiCategoryId is parsed with ast.literal_eval; the first dict's ``name`` becomes
Venue Category Name; Venue Category ID is a stable 24-char hex from url or name.
Missing Timezone Offset is filled with 0 (UTC); UTCTime is rewritten to the
string format required by dataprocess/prepare.py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataprocess.io import EXPECTED_COLUMNS


def category_id_name(raw: str) -> tuple[str, str]:
    s = (raw if isinstance(raw, str) else "").strip()
    if not s:
        return "0" * 24, "Unknown"
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:24], "Unknown"
    if not isinstance(v, list) or not v:
        return "0" * 24, "Unknown"
    d = v[0]
    if not isinstance(d, dict) or "name" not in d:
        return "0" * 24, "Unknown"
    name = str(d["name"])
    key = (d.get("url") or name).strip()
    cid = hashlib.md5(key.encode("utf-8")).hexdigest()[:24]
    return cid, name


def utc_to_prepare_string(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    # Naive local display in UTC; +0000 matches Timezone Offset 0 in prepare
    return ts.dt.tz_convert("UTC").dt.strftime("%a %b %d %H:%M:%S %z %Y")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert CA.csv to Foursquare-style 8-column CSV.")
    p.add_argument("--input", type=Path, default=Path("datasets/CA.csv"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/CA_fsq.csv"),
        help="Default: datasets/CA_fsq.csv (replace datasets/CA.csv when satisfied).",
    )
    p.add_argument("--timezone-offset-minutes", type=int, default=0, help="Fills Timezone Offset; default 0 (UTC).")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, dtype={"UserId": "Int64", "PoiId": "Int64"})
    need = {"UserId", "PoiId", "PoiCategoryId", "Latitude", "Longitude", "UTCTime"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    cats = df["PoiCategoryId"].map(lambda x: category_id_name(x if pd.notna(x) else ""))
    df["Venue Category ID"] = cats.map(lambda t: t[0])
    df["Venue Category Name"] = cats.map(lambda t: t[1])

    out = pd.DataFrame(
        {
            "User ID": df["UserId"].astype(str),
            "Venue ID": df["PoiId"].astype(str),
            "Venue Category ID": df["Venue Category ID"],
            "Venue Category Name": df["Venue Category Name"],
            "Latitude": df["Latitude"].astype(float),
            "Longitude": df["Longitude"].astype(float),
            "Timezone Offset": args.timezone_offset_minutes,
            "UTC Time": utc_to_prepare_string(df["UTCTime"]),
        }
    )
    bad_time = out["UTC Time"].isna().sum()
    if bad_time:
        print(f"Warning: {bad_time} rows had unparseable UTCTime (dropped).", file=sys.stderr)
        out = out.dropna(subset=["UTC Time"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print("Columns:", list(out.columns))
    assert list(out.columns) == EXPECTED_COLUMNS, "Column order must match dataprocess/io.EXPECTED_COLUMNS"


if __name__ == "__main__":
    main()
