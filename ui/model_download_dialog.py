import os
import urllib.request
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
import hashlib

EMB_DIR = "offline_models/embedding"
LLM_DIR = "offline_models/language"

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(LLM_DIR, exist_ok=True)


# -------------------------------
# Worker thread for downloading
# -------------------------------
class DownloadWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, url, dest):
        super().__init__()
        self.url = url
        self.dest = dest

    def run(self):
        def report(blocks, block_size, total_size):
            if total_size > 0:
                percent = int((blocks * block_size * 100) / total_size)
                self.progress.emit(percent)

        urllib.request.urlretrieve(self.url, self.dest, report)
        self.finished.emit(self.dest)


# -------------------------------
# Dialog UI
# -------------------------------
class ModelDownloadDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Download Required Models")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select Language Model:"))
        self.llm_combo = QComboBox()
        self.llm_models = [
            ("Llama-3.2-1B-Instruct-Q4_K_M.gguf",
             "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),

            ("Llama-3.2-1B-Instruct-Q3_K_XL.gguf",
             "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_XL.gguf"),
        ]
        self.llm_combo.addItems([m[0] for m in self.llm_models])
        layout.addWidget(self.llm_combo)

        layout.addWidget(QLabel("Select Embedding Model:"))
        self.emb_combo = QComboBox()
        self.emb_models = [
            ("bge-base-en-v1.5-q8_0.gguf",
             "https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q8_0.gguf"),
            ("bge-base-en-v1.5-q4_k_m.gguf",
             "https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_k_m.gguf"),
        ]
        self.emb_combo.addItems([m[0] for m in self.emb_models])
        layout.addWidget(self.emb_combo)

        self.status = QLabel("")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.download_btn = QPushButton("Download Models")
        self.download_btn.clicked.connect(self.start_download)
        layout.addWidget(self.download_btn)

        self.worker = None
        self.queue = []  # list of (url, dest)

    # -----------------------------------
    # Start downloading the models
    # -----------------------------------
    def start_download(self):
        # Build download queue
        llm_name, llm_url = self.llm_models[self.llm_combo.currentIndex()]
        emb_name, emb_url = self.emb_models[self.emb_combo.currentIndex()]

        llm_dest = os.path.join(LLM_DIR, llm_name)
        emb_dest = os.path.join(EMB_DIR, emb_name)

        if not os.listdir(LLM_DIR):
            self.queue.append((llm_url, llm_dest))
        if not os.listdir(EMB_DIR):
            self.queue.append((emb_url, emb_dest))

        if not self.queue:
            self.accept()
            return

        self.download_next()

    # -----------------------------------
    # Download files sequentially  
    # -----------------------------------
    def download_next(self):
        if not self.queue:
            self.accept()
            return

        url, dest = self.queue.pop(0)
        self.status.setText(f"Downloading: {os.path.basename(dest)}")

        self.worker = DownloadWorker(url, dest)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.download_next)
        self.worker.start()
