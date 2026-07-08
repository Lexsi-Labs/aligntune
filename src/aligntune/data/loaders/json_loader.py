from datasets import load_dataset
from .base import BaseLoader

class JSONLoader(BaseLoader):
    def __init__(self, path: str):
        self.path = path

    def load(self):
        return load_dataset(
            "json",
            data_files=self.path,
        )
