from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QCheckBox, QTabWidget, QWidget, QFileDialog, QListWidget, QMessageBox
)
from rag_engine.config_manager import ConfigManager
from rag_engine.model_manager import list_ollama_models, is_gpu_available

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(600, 500)
        self.config = ConfigManager()
        
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- General Tab ---
        general_tab = QWidget()
        gen_layout = QVBoxLayout(general_tab)
        
        self.gpu_check = QCheckBox("Use GPU (if available)")
        self.gpu_check.setChecked(self.config.get("use_gpu", True))
        if not is_gpu_available():
            self.gpu_check.setEnabled(False)
            self.gpu_check.setText("Use GPU (Not detected)")
        
        gen_layout.addWidget(self.gpu_check)
        gen_layout.addStretch()
        self.tabs.addTab(general_tab, "General")

        # --- API Keys Tab ---
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        
        api_layout.addWidget(QLabel("OpenAI API Key:"))
        self.openai_key = QLineEdit(self.config.get("openai_api_key", ""))
        self.openai_key.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.openai_key)

        api_layout.addWidget(QLabel("Google Gemini API Key:"))
        self.google_key = QLineEdit(self.config.get("google_api_key", ""))
        self.google_key.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.google_key)
        
        api_layout.addStretch()
        self.tabs.addTab(api_tab, "API Keys")

        # --- Models Tab ---
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        model_layout.addWidget(QLabel("Custom Model Paths (.gguf):"))
        self.model_list = QListWidget()
        self.refresh_custom_models()
        model_layout.addWidget(self.model_list)
        
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Model")
        add_btn.clicked.connect(self.add_custom_model)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_custom_model)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        model_layout.addLayout(btn_row)

        model_layout.addWidget(QLabel("Ollama Status:"))
        ollama_models = list_ollama_models()
        if ollama_models:
            status = f"Detected {len(ollama_models)} models: {', '.join(ollama_models[:3])}..."
        else:
            status = "Ollama not detected or no models found."
        model_layout.addWidget(QLabel(status))

        model_layout.addStretch()
        self.tabs.addTab(model_tab, "Models")

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def refresh_custom_models(self):
        self.model_list.clear()
        paths = self.config.get("custom_model_paths", [])
        for p in paths:
            self.model_list.addItem(p)

    def add_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Models (*.gguf)")
        if path:
            current_paths = self.config.get("custom_model_paths", [])
            if path not in current_paths:
                current_paths.append(path)
                self.config.set("custom_model_paths", current_paths)
                self.refresh_custom_models()

    def remove_custom_model(self):
        row = self.model_list.currentRow()
        if row >= 0:
            item = self.model_list.takeItem(row)
            path = item.text()
            current_paths = self.config.get("custom_model_paths", [])
            if path in current_paths:
                current_paths.remove(path)
                self.config.set("custom_model_paths", current_paths)

    def save_settings(self):
        self.config.set("use_gpu", self.gpu_check.isChecked())
        self.config.set("openai_api_key", self.openai_key.text().strip())
        self.config.set("google_api_key", self.google_key.text().strip())
        self.accept()
