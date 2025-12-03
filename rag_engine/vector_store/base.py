from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadatas: Optional[List[dict]] = None):
        """Add documents and their embeddings to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float, dict]]:
        """
        Search for similar documents.
        Returns: List of (document_text, score, metadata)
        """
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Check if the store exists/has data."""
        pass
