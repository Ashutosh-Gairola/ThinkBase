from llama_cpp import Llama
import ollama
import os
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from rag_engine.model_manager import get_embedding_path, is_gpu_available
from rag_engine.config_manager import ConfigManager

class Embedder:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.config = ConfigManager()
        self.llm = None

        if self.provider == "local":
            path = get_embedding_path(self.model_name)
            n_gpu_layers = -1 if is_gpu_available() and self.config.get("use_gpu", True) else 0
            self.llm = Llama(
                model_path=path,
                embedding=True,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
        elif self.provider == "openai":
            if HAS_OPENAI:
                api_key = self.config.get("openai_api_key")
                self.client = openai.OpenAI(api_key=api_key)
        elif self.provider == "google":
            if HAS_GOOGLE:
                api_key = self.config.get("google_api_key")
                genai.configure(api_key=api_key)

    def embed_text(self, text: str):
        if self.provider == "ollama":
            resp = ollama.embed(model=self.model_name, input=text)
            # Ollama returns {'embeddings': [[...]]} for single input
            embeddings = resp.get("embeddings", [])
            if embeddings:
                return embeddings[0]
            return []

        elif self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("OpenAI library not installed")
            resp = self.client.embeddings.create(input=[text], model=self.model_name or "text-embedding-3-small")
            return resp.data[0].embedding

        elif self.provider == "google":
            if not HAS_GOOGLE:
                raise ImportError("Google GenerativeAI library not installed")
            # Google embedding model usually "models/embedding-001"
            model = self.model_name if self.model_name else "models/embedding-001"
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']

        elif self.provider == "local":
            out = self.llm.embed(text)
            # Handle various llama-cpp-python outputs
            if isinstance(out, list) and all(isinstance(x, float) for x in out):
                return out
            if isinstance(out, dict) and "embedding" in out:
                return out["embedding"]
            if isinstance(out, dict) and "data" in out:
                data = out["data"]
                if isinstance(data, list) and "embedding" in data[0]:
                    return data[0]["embedding"]
            # Fallback for token embeddings
            if isinstance(out, dict) and "token_embeddings" in out:
                tokens = out["token_embeddings"]
                if tokens:
                    dim = len(tokens[0])
                    pooled = [0.0] * dim
                    for t in tokens:
                        for i, v in enumerate(t):
                            pooled[i] += v
                    return [x / len(tokens) for x in pooled]
            return []
        
        return []
