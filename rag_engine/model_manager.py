import os
import ollama
try:
    # Optional: Check for CUDA if needed, but for now we rely on config
    pass 
except ImportError:
    pass

from rag_engine.config_manager import ConfigManager

OFFLINE_DIR = os.path.join(os.getcwd(), "offline_models")
EMB_DIR = os.path.join(OFFLINE_DIR, "embedding")
LLM_DIR = os.path.join(OFFLINE_DIR, "language")

# Ensure directories exist
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(LLM_DIR, exist_ok=True)

def is_gpu_available():
    # We assume GPU is available if the user wants to use it.
    # llama-cpp-python handles fallback or errors if configured incorrectly.
    return True

def list_local_embedding_models():
    return [
        f for f in os.listdir(EMB_DIR)
        if f.lower().endswith(".gguf")
    ]

def list_local_llm_models():
    return [
        f for f in os.listdir(LLM_DIR)
        if f.lower().endswith(".gguf")
    ]

def list_ollama_models():
    try:
        models = ollama.list()
        # ollama.list() returns a dict with 'models' key which is a list of dicts
        if isinstance(models, dict) and 'models' in models:
             return [m['name'] for m in models['models']]
        return []
    except Exception:
        return []

def list_custom_models():
    config = ConfigManager()
    paths = config.get("custom_model_paths", [])
    valid_models = []
    for p in paths:
        if os.path.exists(p) and p.lower().endswith(".gguf"):
            valid_models.append(p)
    return valid_models

def get_embedding_path(model_name: str):
    # Check if it's a full path (custom model)
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name
    return os.path.join(EMB_DIR, model_name)

def get_llm_path(model_name: str):
    # Check if it's a full path (custom model)
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name
    return os.path.join(LLM_DIR, model_name)
