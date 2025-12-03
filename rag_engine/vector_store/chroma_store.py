import chromadb
import os
from typing import List, Tuple, Optional
from .base import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str, persist_directory: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadatas: Optional[List[dict]] = None):
        if not documents:
            return
            
        ids = [str(i) for i in range(len(documents))] # Simple ID generation for now, or use UUIDs
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
            
        # Chroma expects ids
        # Check if we need to generate unique IDs or if we can just append.
        # For simplicity in this refactor, we might want to clear and re-add or handle IDs better.
        # But for "add_documents" usually we assume new docs. 
        # Let's generate UUIDs to be safe.
        import uuid
        ids = [str(uuid.uuid4()) for _ in documents]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float, dict]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # results is a dict with lists of lists
        if not results['documents']:
            return []
            
        # Format: [(doc, score, metadata), ...]
        # Chroma returns distances by default, so lower is better. 
        # But our interface might expect similarity (higher is better).
        # We should clarify or just return what Chroma gives and handle it.
        # Usually RAG engines want similarity. Chroma can do cosine similarity.
        # Let's assume default for now (L2 distance) and maybe convert if needed.
        # Or just return the distance and let the engine handle it.
        
        docs = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        output = []
        for doc, dist, meta in zip(docs, distances, metadatas):
            # Convert distance to similarity score if possible, or just return distance
            # For L2, similarity = 1 / (1 + distance) is a common approx
            score = 1.0 / (1.0 + dist) 
            output.append((doc, score, meta))
            
        return output

    def exists(self) -> bool:
        return self.collection.count() > 0
    
    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
