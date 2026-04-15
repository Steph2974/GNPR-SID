from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataprocess.config import DatasetConfig
from dataprocess.mapping import map_ids
from dataprocess.paths import DataLayout
from dataprocess.poi_info import build_poi_info
from dataprocess.prepare import prepare_and_filter_raw_dataset
from dataprocess.sequences import build_sequence_datasets
from dataprocess.split import time_split_train_test
from dataprocess.stats import history_length_stats


def run(cfg: DatasetConfig) -> None:
    dataset = cfg.dataset
    layout = DataLayout.resolve(cfg.data_root, cfg.output_root)

    # 1) raw -> filtered (Region = Plus Code prefix)
    filtered_csv = prepare_and_filter_raw_dataset(
        dataset,
        cfg.poi_min_freq,
        cfg.user_min_freq,
        raw_csv=layout.raw_csv(dataset),
        out_filtered_csv=layout.filtered_csv(dataset),
    )

    # 2) map ids -> {output_root}/{dataset}/data.csv (+ mapping csvs)
    map_ids(
        dataset=dataset,
        input_csv=filtered_csv,
        out_dir=layout.dataset_dir(dataset),
        seed=0,
    )

    data_csv = layout.data_csv(dataset)

    # 3) poi_info.csv for embedding dataset
    build_poi_info(dataset, data_csv=data_csv, out_path=layout.poi_info_csv(dataset))

    # 4) train/test split (expanded test histories)
    time_split_train_test(
        data_csv=data_csv,
        out_train_csv=layout.train_data_csv(dataset),
        out_test_csv=layout.test_data_csv(dataset),
        train_ratio=0.8,
    )

    # 5) sequence datasets -> {output_root}/{dataset}/data/{train,val,test,test_all}.csv
    build_sequence_datasets(
        dataset=dataset,
        train_data_csv=layout.train_data_csv(dataset),
        test_data_csv=layout.test_data_csv(dataset),
        all_data_csv=data_csv,
        out_dir=layout.sequences_dir(dataset),
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        mask_prob=cfg.mask_prob,
        seed=0,
    )

    # 6) stats
    st = history_length_stats(data_csv)
    print(f"[{dataset}] average_history_length={st.average_history_length:.6f} most_frequent_length={st.most_frequent_length}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Modularized dataprocess-v2.1 pipeline.")
    p.add_argument("--dataset", type=str, default="NYC", help="Dataset name, e.g. NYC/TKY/CA")
    p.add_argument(
        "--data-root",
        type=str,
        default="datasets",
        help="Directory containing raw ``{dataset}.csv`` (default: datasets).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory for all generated files under ``{dataset}/``; default: same as --data-root.",
    )
    p.add_argument("--poi_min_freq", type=int, default=10)
    p.add_argument("--user_min_freq", type=int, default=10)
    p.add_argument("--window_size", type=int, default=50)
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--mask_prob", type=float, default=0.1)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = DatasetConfig(
        dataset=args.dataset,
        data_root=Path(args.data_root),
        output_root=Path(args.output_root) if args.output_root else None,
        poi_min_freq=args.poi_min_freq,
        user_min_freq=args.user_min_freq,
        window_size=args.window_size,
        step_size=args.step_size,
        mask_prob=args.mask_prob,
    )
    run(cfg)


if __name__ == "__main__":
    main()

    # python scripts/dataprocess_v2_1.py --dataset NYC --output-root ./datasets

