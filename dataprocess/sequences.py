from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io import ensure_dir, write_csv


@dataclass(frozen=True)
class SequenceOutputs:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    test_all: pd.DataFrame


def generate_train_sequences(df: pd.DataFrame, window_size: int, step_size: int, mask_prob: float, *, seed: int | None = 0) -> pd.DataFrame:
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])

    rng = random.Random(seed)
    results: list[dict] = []

    for uid, group in df.groupby("Uid"):
        group = group.sort_values("Time").reset_index(drop=True)
        if len(group) > 80:
            group = group.iloc[-80:]

        n = len(group)

        if n < window_size:
            if n >= 10:
                results.append(
                    {
                        "Uid": uid,
                        "Pids": group["Pid"].iloc[:-1].tolist(),
                        "Times": group["Time"].iloc[:-1].tolist(),
                        "Target": group["Pid"].iloc[-1],
                        "Target_time": group["Time"].iloc[-1],
                    }
                )
            continue

        for start in range(n - 1, window_size - 2, -step_size):
            window = group.iloc[start - window_size + 1 : start + 1]

            input_pids = window["Pid"].iloc[:-1].tolist()
            input_times = window["Time"].iloc[:-1].tolist()
            original_target_pid = window["Pid"].iloc[-1]
            original_target_time = window["Time"].iloc[-1]

            if rng.random() < mask_prob and len(input_pids) >= 1:
                drop_idx = rng.randint(0, len(input_pids) - 1)
                target_pid = input_pids[drop_idx]
                target_time = input_times[drop_idx]
                input_pids = input_pids[:drop_idx] + input_pids[drop_idx + 1 :] + [original_target_pid]
                input_times = input_times[:drop_idx] + input_times[drop_idx + 1 :] + [original_target_time]
            else:
                target_pid = original_target_pid
                target_time = original_target_time

            results.append(
                {
                    "Uid": uid,
                    "Pids": input_pids,
                    "Times": input_times,
                    "Target": target_pid,
                    "Target_time": target_time,
                }
            )

    train = pd.DataFrame(results)
    train["Times"] = train["Times"].apply(lambda xs: [t.strftime("%Y-%m-%d %H:%M") for t in xs])
    train["Target_time"] = pd.to_datetime(train["Target_time"]).dt.strftime("%Y-%m-%d %H:%M")
    return train


def generate_test_sequences(df: pd.DataFrame, window_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])

    val_records: list[dict] = []
    test_records: list[dict] = []

    for uid, group in df.groupby("Uid"):
        group = group.sort_values("Time").reset_index(drop=True)
        n = len(group)

        if n < window_size:
            if n > 2:
                test_records.append(
                    {
                        "Uid": uid,
                        "Pids": group["Pid"].iloc[:-1].tolist(),
                        "Times": group["Time"].iloc[:-1].tolist(),
                        "Target": group["Pid"].iloc[-1],
                        "Target_time": group["Time"].iloc[-1],
                    }
                )
                val_records.append(
                    {
                        "Uid": uid,
                        "Pids": group["Pid"].iloc[:-2].tolist(),
                        "Times": group["Time"].iloc[:-2].tolist(),
                        "Target": group["Pid"].iloc[-2],
                        "Target_time": group["Time"].iloc[-2],
                    }
                )
            continue

        if n >= window_size + 1:
            val_start = n - window_size - 1
            val_window = group.iloc[val_start : val_start + window_size]
            val_records.append(
                {
                    "Uid": uid,
                    "Pids": val_window["Pid"].iloc[:-1].tolist(),
                    "Times": val_window["Time"].iloc[:-1].tolist(),
                    "Target": val_window["Pid"].iloc[-1],
                    "Target_time": val_window["Time"].iloc[-1],
                }
            )

        test_window = group.iloc[n - window_size :]
        test_records.append(
            {
                "Uid": uid,
                "Pids": test_window["Pid"].iloc[:-1].tolist(),
                "Times": test_window["Time"].iloc[:-1].tolist(),
                "Target": test_window["Pid"].iloc[-1],
                "Target_time": test_window["Time"].iloc[-1],
            }
        )

    val_df = pd.DataFrame(val_records)
    test_df = pd.DataFrame(test_records)

    for d in (val_df, test_df):
        d["Times"] = d["Times"].apply(lambda xs: [t.strftime("%Y-%m-%d %H:%M") for t in xs])
        d["Target_time"] = pd.to_datetime(d["Target_time"]).dt.strftime("%Y-%m-%d %H:%M")

    return val_df, test_df


def build_sequence_datasets(
    dataset: str,
    train_data_csv: str | Path,
    test_data_csv: str | Path,
    all_data_csv: str | Path,
    out_dir: str | Path,
    *,
    window_size: int,
    step_size: int,
    mask_prob: float,
    seed: int | None = 0,
) -> SequenceOutputs:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    train_raw = pd.read_csv(train_data_csv)
    test_raw = pd.read_csv(test_data_csv)
    all_raw = pd.read_csv(all_data_csv)

    train = generate_train_sequences(train_raw, window_size, step_size, mask_prob, seed=seed)
    val, test = generate_test_sequences(test_raw, window_size)
    _, test_all = generate_test_sequences(all_raw, window_size)

    write_csv(train, out_dir / "train.csv")
    write_csv(val, out_dir / "val.csv")
    write_csv(test, out_dir / "test.csv")
    write_csv(test_all, out_dir / "test_all.csv")

    return SequenceOutputs(train=train, val=val, test=test, test_all=test_all)

