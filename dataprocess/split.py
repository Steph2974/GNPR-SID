from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import write_csv


def _remove_unseen_users_pois(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    users_train = df_train["Uid"].unique()
    pois_train = df_train["Pid"].unique()
    df_test = df_test[df_test["Uid"].isin(users_train)]
    df_test = df_test[df_test["Pid"].isin(pois_train)]
    return df_test


def time_split_train_test(data_csv: str | Path, out_train_csv: str | Path, out_test_csv: str | Path, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mimic notebook cell4:
    - keep columns [Uid, Pid, Time]
    - sort by Time (parsed datetime)
    - split by global ratio
    - drop test rows with unseen users/pois vs train
    - 'test_data.csv' is expanded to include full histories of users that appear in test.
    """
    df = pd.read_csv(data_csv)[["Uid", "Pid", "Time"]]
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(by="Time").reset_index(drop=True)

    train_size = int(train_ratio * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    test_df = _remove_unseen_users_pois(train_df, test_df)

    test_uids = test_df["Uid"].unique()
    expanded_df = df[df["Uid"].isin(test_uids)].copy()

    write_csv(train_df, out_train_csv)
    write_csv(expanded_df, out_test_csv)
    return train_df, expanded_df

