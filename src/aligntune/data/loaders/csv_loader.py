from datasets import load_dataset
from .base import BaseLoader

class CSVLoader(BaseLoader):
    def __init__(self, path: str, delimiter: str | None = None):
        self.path = path
        self.delimiter = delimiter

    def load(self):
        if self.delimiter:
            return load_dataset(
                "csv",
                data_files=self.path,
                delimiter=self.delimiter,
            )

        return load_dataset(
            "csv",
            data_files=self.path,
        )
