from abc import ABC, abstractmethod

from .filters import MetadataFilters
from .schema import Document, RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        raise NotImplementedError