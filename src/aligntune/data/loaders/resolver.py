from pathlib import Path
from typing import Any
from .hf_loader import HFLoader
from .json_loader import JSONLoader
from .csv_loader import CSVLoader
from .parquet_loader import ParquetLoader
from .directory_loader import DirectoryLoader

class LoaderResolver:
    @staticmethod
    def resolve(source: str, **kwargs: Any):
        """
        Determines the correct loader based on the source string.
        Passes **kwargs (like config_name, delimiter) to the specific loader.
        """
        path = Path(source)

        if path.exists():
            if path.is_dir():
                # Directory loader usually doesn't need config_name, but might need others
                return DirectoryLoader(source) 
            if source.endswith(".json") or source.endswith(".jsonl"):
                return JSONLoader(source)
            if source.endswith(".csv"):
                # CSVLoader needs 'delimiter' which might be in kwargs
                delimiter = kwargs.get("delimiter")
                return CSVLoader(source, delimiter=delimiter)
            if source.endswith(".parquet"):
                return ParquetLoader(source)

        # Default to HuggingFace
        # We pass all kwargs here (config_name, split, cache_dir, etc.)
        return HFLoader(source, **kwargs)