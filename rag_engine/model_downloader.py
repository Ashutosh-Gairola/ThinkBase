import os
import urllib.request


OFFLINE_DIR = os.path.join(os.getcwd(), "offline_models")
EMB_DIR = os.path.join(OFFLINE_DIR, "embedding")
LLM_DIR = os.path.join(OFFLINE_DIR, "language")

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(LLM_DIR, exist_ok=True)


# ------------------------------
# Default models to download
# ------------------------------

# Priority: Q4_K_M → smaller & fast
LANGUAGE_MODELS = [
    (
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    (
        "Llama-3.2-1B-Instruct-Q3_K_XL.gguf",
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_XL.gguf"
    )
]

EMBEDDING_MODELS = [
    (
        "bge-base-en-v1.5-q8_0.gguf",
        "https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q8_0.gguf"
    ),
    (
        "bge-base-en-v1.5-q4_k_m.gguf",
        "https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_k_m.gguf"
    )
]


# ------------------------------
# Helper: download with progress
# ------------------------------
def download_with_progress(url, dest_path):
    print(f"Downloading:\n{url}\n→ {dest_path}")

    def report(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgress: {percent}%", end="")

    urllib.request.urlretrieve(url, dest_path, report)
    print("\nDownload complete!")


# ------------------------------
# Check & download logic
# ------------------------------
def ensure_models_present():
    # ---- Check embedding folder ----
    existing_emb = [f for f in os.listdir(EMB_DIR) if f.endswith(".gguf")]
    if existing_emb:
        print("Embedding model already exists:", existing_emb)
    else:
        # Download first available embedding model
        filename, url = EMBEDDING_MODELS[0]
        dest = os.path.join(EMB_DIR, filename)
        download_with_progress(url, dest)

    # ---- Check language folder ----
    existing_llm = [f for f in os.listdir(LLM_DIR) if f.endswith(".gguf")]
    if existing_llm:
        print("Language model already exists:", existing_llm)
    else:
        # Download first available LLM
        filename, url = LANGUAGE_MODELS[0]
        dest = os.path.join(LLM_DIR, filename)
        download_with_progress(url, dest)

    print("✓ Model check completed.")
