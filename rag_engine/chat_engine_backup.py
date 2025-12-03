import ollama
from llama_cpp import Llama
import os
import json

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

from rag_engine.model_manager import get_llm_path, is_gpu_available
from rag_engine.config_manager import ConfigManager
from rag_engine.history_manager import HistoryManager

class ChatEngine:
    def __init__(self, vector_store, embedder, provider: str, model_name: str, chat_id: str = None):
        self.vs = vector_store
        self.embedder = embedder
        self.provider = provider
        self.model_name = model_name
        self.config = ConfigManager()
        self.history_manager = HistoryManager()
        self.chat_id = chat_id
        
        self.llm_local = None
        self.openai_client = None

        if self.provider == "local":
            path = get_llm_path(self.model_name)
            n_gpu_layers = -1 if is_gpu_available() and self.config.get("use_gpu", True) else 0
            self.llm_local = Llama(
                model_path=path,
                n_ctx=4096,
                n_threads=4,
                n_gpu_layers=n_gpu_layers,
                chat_format="llama-3", # Default, can be configurable
                verbose=False
            )
        elif self.provider == "openai":
            if HAS_OPENAI:
                api_key = self.config.get("openai_api_key")
                self.openai_client = openai.OpenAI(api_key=api_key)
        elif self.provider == "google":
            if HAS_GOOGLE:
                api_key = self.config.get("google_api_key")
                genai.configure(api_key=api_key)
                self.google_model = genai.GenerativeModel(self.model_name or "gemini-pro")

    def build_instruction(self, retrieved_context):
        print(retrieved_context)
        return f"""You are a helpful assistant. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{retrieved_context}
"""

    def retrieve(self, query):
        # Embed query
        query_vec = self.embedder.embed_text(query)
        # Search in vector store
        results = self.vs.search(query_vec, top_k=3)
        return "\n\n".join([r[0] for r in results])

    def chat(self, query, retrieved, stream=True):
        system_prompt = self.build_instruction(retrieved)
        
        # We don't strictly need to pass history to the model API if we want to save tokens,
        # but for a real chat experience we should. 
        # For now, let's just use the current query + system prompt with context.
        # If full history is needed, we would load it from HistoryManager.

        if self.provider == "ollama":
            resp = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                stream=stream
            )
            if stream:
                return resp # generator
            return resp["message"]["content"]

        elif self.provider == "local":
            resp = self.llm_local.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                stream=stream
            )
            if stream:
                # Adapter for local stream to match expected format
                def local_stream_adapter():
                    for chunk in resp:
                        if not chunk.get("choices"):
                            continue
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            yield {"message": {"content": delta["content"]}}
                return local_stream_adapter()
            return resp["choices"][0]["message"]["content"]

        elif self.provider == "openai":
            if not self.openai_client:
                yield {"message": {"content": "OpenAI API Key not configured."}}
                return

            resp = self.openai_client.chat.completions.create(
                model=self.model_name or "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                stream=stream
            )
            if stream:
                def openai_stream_adapter():
                    for chunk in resp:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield {"message": {"content": content}}
                return openai_stream_adapter()
            return resp.choices[0].message.content

        elif self.provider == "google":
            if not HAS_GOOGLE:
                yield {"message": {"content": "Google API Key not configured."}}
                return
            
            # Google Generative AI chat
            chat = self.google_model.start_chat(history=[])
            # We inject system prompt into the first message or as context if possible, 
            # but Gemini API is a bit different. Let's just prepend context to query.
            full_query = f"{system_prompt}\n\nUser Question: {query}"
            
            response = chat.send_message(full_query, stream=stream)
            
            if stream:
                def google_stream_adapter():
                    for chunk in response:
                        if chunk.text:
                            yield {"message": {"content": chunk.text}}
                return google_stream_adapter()
            return response.text

        return "Provider not supported"

    def save_history(self, history):
        if self.chat_id:
            # Load existing to preserve metadata
            existing = self.history_manager.load_chat(self.chat_id)
            if existing:
                existing["messages"] = history
                self.history_manager.save_chat(existing)

    def load_history(self):
        if self.chat_id:
            chat = self.history_manager.load_chat(self.chat_id)
            if chat:
                return chat.get("messages", [])
        return []