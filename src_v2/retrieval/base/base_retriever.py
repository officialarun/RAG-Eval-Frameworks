from abc import (
    ABC,
    abstractmethod
)

from src_v2.models import (
    RetrievalResult
)


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int
    ) -> list[RetrievalResult]:

        pass