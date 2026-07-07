from abc import (
    ABC,
    abstractmethod
)

from raglens.models import (
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