from abc import ABC, abstractmethod

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, query: str) -> str:
        pass

class LocalKnowledgeTool(BaseTool):
    def __init__(self, vector_store, embedder):
        super().__init__(
            name="search_knowledge_base",
            description="Useful for answering questions about the specific knowledge base or documents provided. input should be a specific search query."
        )
        self.vs = vector_store
        self.embedder = embedder

    def run(self, query: str) -> str:
        # Embed query
        query_vec = self.embedder.embed_text(query)
        
        # Search in vector store
        # Returns [(doc, score, metadata)]
        results = self.vs.search(query_vec, top_k=3)
        
        if not results:
            return "No relevant information found in the knowledge base."
            
        # Format results
        formatted = []
        for i, (doc, score, meta) in enumerate(results, 1):
            formatted.append(f"Result {i} (Score: {score:.4f}):\n{doc}")
            
        return "\n\n".join(formatted)
