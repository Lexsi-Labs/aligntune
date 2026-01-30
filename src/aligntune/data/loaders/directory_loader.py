from pathlib import Path
from datasets import DatasetDict
from .json_loader import JSONLoader
from .csv_loader import CSVLoader
from .parquet_loader import ParquetLoader
from .base import BaseLoader

class DirectoryLoader(BaseLoader):
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self):
        files = list(self.path.glob("*"))
        datasets = {}

        for f in files:
            if f.suffix in [".json", ".jsonl"]:
                datasets[f.stem] = JSONLoader(str(f)).load()["train"]
            elif f.suffix == ".csv":
                datasets[f.stem] = CSVLoader(str(f)).load()["train"]
            elif f.suffix == ".parquet":
                datasets[f.stem] = ParquetLoader(str(f)).load()["train"]

        if not datasets:
            raise ValueError(f"No supported files found in {self.path}")

        return DatasetDict(datasets)
