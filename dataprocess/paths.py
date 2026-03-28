from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLayout:
    """
    Layout for pipeline I/O.

    - ``data_root``: directory that contains the raw ``{dataset}.csv`` (e.g. ``datasets/NYC.csv``).
    - ``output_root``: directory under which all generated files are written
      (``output_root/{dataset}/data.csv``, mappings, ``data/train.csv``, etc.).
      If omitted at construction time, it defaults to ``data_root``.
    """

    data_root: Path
    output_root: Path

    @staticmethod
    def resolve(data_root: Path | str = "datasets", output_root: Path | str | None = None) -> DataLayout:
        dr = Path(data_root)
        oroot = Path(output_root) if output_root is not None else dr
        return DataLayout(data_root=dr, output_root=oroot)

    def raw_csv(self, dataset: str) -> Path:
        return self.data_root / f"{dataset}.csv"

    def dataset_dir(self, dataset: str) -> Path:
        return self.output_root / dataset

    def filtered_csv(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / f"{dataset}.csv"

    def data_csv(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "data.csv"

    def poi_info_csv(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "poi_info.csv"

    def train_data_csv(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "train_data.csv"

    def test_data_csv(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "test_data.csv"

    def sequences_dir(self, dataset: str) -> Path:
        return self.dataset_dir(dataset) / "data"
