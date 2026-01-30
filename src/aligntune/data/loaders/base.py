from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> Dataset | DatasetDict:
        pass
