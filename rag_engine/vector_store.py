"""
Vector store using pickle per-document.
VECTOR_DB structure: list of tuples -> (chunk_text, embedding_list)
"""

import os
import pickle
from typing import List, Tuple
import numpy as np

EMBED_DIR = os.path.join("data", "embeddings")
os.makedirs(EMBED_DIR, exist_ok=True)

class VectorStore:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.filepath = os.path.join(EMBED_DIR, f"{doc_id}.pkl")
        self.vectors: List[Tuple[str, List[float]]] = []

    def exists(self) -> bool:
        return os.path.exists(self.filepath)

    def save(self):
        with open(self.filepath, "wb") as f:
            pickle.dump(self.vectors, f)

    def load(self):
        if not self.exists():
            return False
        with open(self.filepath, "rb") as f:
            self.vectors = pickle.load(f)
        return True

    def build_from_chunks(self, chunks: list, embedder, progress_callback=None):
        """
        chunks: list[str]
        embedder: Embedder instance with embed_text()
        progress_callback: optional callable(current, total)
        """
        self.vectors = []
        total = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            emb = embedder.embed_text(chunk)
            self.vectors.append((chunk, emb))
            if progress_callback:
                try:
                    progress_callback(i, total)
                except Exception:
                    pass
        # save after build
        self.save()

    def _cosine_similarity(self, query: np.ndarray, vec: np.ndarray) -> float:
        """Calculate cosine similarity between query and vector."""
        norm_q = np.linalg.norm(query)
        norm_v = np.linalg.norm(vec)
        if norm_q == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(query, vec) / (norm_q * norm_v))

    def _euclidean_similarity(self, query: np.ndarray, vec: np.ndarray) -> float:
        """Calculate similarity using negative euclidean distance (higher is better)."""
        distance = np.linalg.norm(query - vec)
        # Convert distance to similarity (inverse, normalized)
        # Using 1 / (1 + distance) to ensure positive similarity scores
        return float(1.0 / (1.0 + distance))

    def _dot_product_similarity(self, query: np.ndarray, vec: np.ndarray) -> float:
        """Calculate dot product similarity."""
        return float(np.dot(query, vec))

    def _manhattan_similarity(self, query: np.ndarray, vec: np.ndarray) -> float:
        """Calculate similarity using negative manhattan distance (higher is better)."""
        distance = np.sum(np.abs(query - vec))
        # Convert distance to similarity
        return float(1.0 / (1.0 + distance))

    def search(self, query_vec: List[float], top_k: int = 3, similarity_method: str = "cosine"):
        """
        Search for similar vectors.
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of top results to return
            similarity_method: One of "cosine", "euclidean", "dot_product", "manhattan"
        
        Returns:
            List of tuples: [(chunk_text, similarity_score), ...]
        """
        if not self.vectors:
            return []
            
        query = np.array(query_vec)
        
        # Select similarity function
        similarity_funcs = {
            "cosine": self._cosine_similarity,
            "euclidean": self._euclidean_similarity,
            "dot_product": self._dot_product_similarity,
            "manhattan": self._manhattan_similarity
        }
        
        similarity_func = similarity_funcs.get(similarity_method.lower(), self._cosine_similarity)
            
        scores = []
        for chunk, emb in self.vectors:
            vec = np.array(emb)
            score = similarity_func(query, vec)
            scores.append((score, chunk))
            
        # Sort by score (descending - higher is better)
        scores.sort(key=lambda x: x[0], reverse=True)
        return [(s[1], s[0]) for s in scores[:top_k]]

    def get_all(self):
        return self.vectors
