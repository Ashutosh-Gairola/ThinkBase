<details>
<summary><strong>⚠️ IMPORTANT — Developer Note</strong></summary>
<br>

ThinkBase is a **trial / experimental project** created as part of my personal learning journey.

I am **new to developing Windows desktop applications**, packaging Python apps, and working with offline LLM pipelines.  
There may be limitations or missing optimizations, and I am continuously learning and improving the project over time.

Your feedback, suggestions, and contributions are greatly appreciated.

Thank you for trying ThinkBase!


</details>

# 📘 ThinkBase — Private Offline RAG Desktop App

ThinkBase is a fully offline, privacy-focused RAG (Retrieval Augmented Generation) desktop application built using Python, llama.cpp, PySide6, and Tesseract OCR.

ThinkBase allows you to:

- Upload PDF, DOCX, and TXT files
- Extract + clean + OCR text
- Generate embeddings locally
- Store vector embeddings
- Retrieve relevant chunks
- Chat with a local LLM using document-aware context

**No internet connection is required.**  
**No cloud APIs are used.**  
**Your data never leaves your device.**

## ✨ Features

### 🔍 Fully Offline RAG Pipeline

ThinkBase performs all processing locally:

- Embedding generation
- Document retrieval
- LLM chat inference
- OCR for scanned PDFs

Supports offline GGUF models for both embedding and LLM chat.

### 🧠 Local Embeddings via llama.cpp

Uses BGE GGUF models such as:

- `bge-base-en-v1.5-q4_k_m.gguf`
- `bge-base-en-v1.5-q8_0.gguf`

Embedding generation is fast, accurate, and completely offline.

### 💬 Local LLM Chat

ThinkBase runs your chosen instruction model locally:

- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B`

Any GGUF-based instruct model

Context is automatically constructed from retrieved document chunks.

### 🔎 OCR for Scanned PDFs

- Bundled portable Tesseract OCR
- Automatically detects scanned pages
- Extracts text and cleans noise
- No installation required on user machines

### 🖥 Modern GUI (PySide6)

Features:

- Chat bubbles interface
- File upload panel
- Model selectors (LLM + Embeddings)
- Document list with auto-restore chat history
- Streaming LLM responses
- Background worker threads for embedding

### 📂 Persistent Local Storage

- Embeddings saved under: `data/embeddings/`
- Chat history saved under: `data/chat_history/`

Using simple pickle storage (fast & lightweight)

## 📁 Project Structure

```
ThinkBase/
│
├── rag_engine/
│   ├── embedder.py          # llama.cpp embedding wrapper
│   ├── vector_store.py      # Embedding persistence & retrieval
│   ├── processor.py         # PDF/DOCX/TXT loader + OCR + cleaning
│   ├── chat_engine.py       # Chat with model via context injection
│   ├── model_manager.py     # Handles offline + Ollama model detection
│
├── ui/
│   ├── app_window.py        # Main PySide6 GUI
│   ├── chat_bubble.py       # Chat bubble widget
│
├── offline_models/
│   ├── embedding/           # BGE embedding models (GGUF)
│   └── language/            # Llama or other instruct models (GGUF)
│
├── tesseract/               # Portable OCR (exe + tessdata)
│   ├── tesseract.exe
│   └── tessdata/
│
├── data/
│   ├── embeddings/          # Vectorstore .pkl files
│   └── chat_history/        # Per-document chat logs
│
├── assets/
│   └── logo.ico
│
├── main.py                  # App entry point
├── requirements.txt
└── README.md
```

## 🏗 How It Works Internally

User uploads a document

ThinkBase extracts text (OCR if needed)

Text is cleaned and chunked

Embeddings are generated locally via llama.cpp

Embeddings stored in vectorstore

During chat, relevant chunks are retrieved

A custom system prompt is created

Local LLM generates the answer

Chat history is saved per document

## 🛠 Build Standalone EXE (Windows)

ThinkBase supports a full offline build using PyInstaller, including:

- Offline GGUF models
- Portable Tesseract OCR
- llama_cpp backend libraries

#### PowerShell Build Command

From inside the project root:

```powershell
pyinstaller `
    --onefile `
    --noconsole `
    --icon=assets/logo.ico `
    --name=ThinkBase `
    --add-data "tesseract;tesseract" `
    --add-data "offline_models;offline_models" `
    --add-data "venv/Lib/site-packages/llama_cpp/lib;llama_cpp/lib" `
    main.py
```

This produces:

`dist/ThinkBase.exe`

A fully portable, fully offline, one-file executable.

## 🔌 Hybrid Mode (Offline + Optional Ollama)

ThinkBase can run in 3 ways:

- ✔ Offline-only (default)

  Loads GGUF models from `offline_models/`.

- ✔ Ollama-only

  If Ollama installed, user may choose Ollama models.

- ✔ Mixed mode

  Example:  
  Offline embeddings + Ollama LLM  
  or  
  Ollama embeddings + offline LLM

The app auto-detects what is available.

## 🔐 Privacy & Security

ThinkBase is designed for confidential documents:

**NO cloud access**  
**NO telemetry**  
**NO network calls**

Everything runs locally

All data stays on disk

Safe for legal docs, HR files, medical documents, enterprise data.

## 📈 Roadmap

- GPU acceleration for llama.cpp
- FAISS support for large-scale retrieval
- Multi-document chat sessions
- Exportable conversation summaries
- Plugin system for automation

## 🤝 Contributing

Pull requests and feature requests are welcome.  
Please open an issue to discuss major changes.
