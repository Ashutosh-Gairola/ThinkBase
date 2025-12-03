import os
import sys
import threading
import uuid
from PySide6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QListWidget, QLabel, QFileDialog,
    QMessageBox, QComboBox, QScrollArea, QFrame, QSplitter, QStackedWidget,
    QToolButton, QMenu
)
from PySide6.QtCore import Qt, Signal, QObject, QSize
from PySide6.QtGui import QFont, QIcon, QAction

from rag_engine.config_manager import ConfigManager
from rag_engine.model_manager import (
    list_local_embedding_models, list_local_llm_models,
    list_ollama_models, list_custom_models
)
from rag_engine.processor import load_document, chunk_text, sanitize_doc_id
from rag_engine.embedder import Embedder
from rag_engine.vector_store import VectorStore
from rag_engine.chat_engine import ChatEngine
from rag_engine.history_manager import HistoryManager

from .chat_bubble import ChatBubble, ContextBubble
from .settings_dialog import SettingsDialog
from .styles import DARK_THEME

# ---------------------------
# Signals for UI thread-safe ops
# ---------------------------
class UISignals(QObject):
    append_bubble = Signal(object)   # expects ChatBubble instance
    typing_update = Signal(str)      # incremental text append to current bubble
    set_status = Signal(str)
    clear_status = Signal()
    enable_input = Signal(bool)
    run_in_main = Signal(object)     # callable to run in main thread
    refresh_history = Signal()       # signal to refresh history list

# ---------------------------
# Embed worker
# ---------------------------
class EmbedWorker(threading.Thread):
    def __init__(self, path, doc_id, signals, embed_provider, embed_model):
        super().__init__(daemon=True)
        self.path = path
        self.doc_id = doc_id
        self.signals = signals
        self.embed_provider = embed_provider
        self.embed_model = embed_model

    def run(self):
        try:
            text = load_document(self.path)
            chunks = chunk_text(text, 300)
            embedder = Embedder(self.embed_provider, self.embed_model)
            vs = VectorStore(self.doc_id)

            def progress(current, total):
                self.signals.set_status.emit(f"Embedding {current}/{total}...")

            vs.build_from_chunks(chunks, embedder, progress_callback=progress)

            self.signals.clear_status.emit()
        except Exception as e:
            self.signals.set_status.emit(f"[Error] {e}")

# ---------------------------
# MainWindow
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Private RAG App 2.0")
        self.resize(1200, 800)
        self.setStyleSheet(DARK_THEME)

        self.config = ConfigManager()
        self.history_manager = HistoryManager()

        # Signals
        self.ui = UISignals(self)
        self.ui.append_bubble.connect(self._slot_append_bubble)
        self.ui.typing_update.connect(self._slot_typing_update)
        self.ui.set_status.connect(self._slot_set_status)
        self.ui.clear_status.connect(self._slot_clear_status)
        self.ui.enable_input.connect(self._slot_enable_input)
        self.ui.run_in_main.connect(self._slot_run_in_main)
        self.ui.refresh_history.connect(self.refresh_chat_history)

        # Main Layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar (Left) ---
        sidebar = QWidget()
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border-right: 2px solid #333;
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)
        
        # App Title
        title_lbl = QLabel("🤖 RAG Chat")
        title_lbl.setObjectName("header")
        title_lbl.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_lbl)
        
        # New Chat Button
        self.new_chat_btn = QPushButton("+ New Chat")
        self.new_chat_btn.setObjectName("primary")
        self.new_chat_btn.clicked.connect(self.on_new_chat)
        sidebar_layout.addWidget(self.new_chat_btn)

        # History List
        history_label = QLabel("📜 History")
        history_label.setStyleSheet("font-weight: bold; color: #cccccc; padding: 5px;")
        sidebar_layout.addWidget(history_label)
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_select)
        sidebar_layout.addWidget(self.history_list)

        # Settings Button (Bottom)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        sidebar_layout.addWidget(self.settings_btn)

        main_layout.addWidget(sidebar)

        # --- Main Content (Center + Right) ---
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Chat Area (Center)
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        # Chat Header
        self.chat_header = QLabel("💬 New Chat")
        self.chat_header.setStyleSheet("""
            QLabel {
                padding: 12px 16px;
                font-weight: bold;
                font-size: 16px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2d2d2d, stop:1 #252525);
                border-bottom: 2px solid #0078d4;
                color: #ffffff;
            }
        """)
        chat_layout.addWidget(self.chat_header)

        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.chat_container = QWidget()
        self.chat_container_layout = QVBoxLayout(self.chat_container)
        self.chat_container_layout.setContentsMargins(20, 20, 20, 20)
        self.chat_container_layout.setSpacing(15)
        self.chat_container_layout.addStretch()
        self.scroll.setWidget(self.chat_container)
        chat_layout.addWidget(self.scroll)

        # Input Area
        input_container = QWidget()
        input_container.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border-top: 2px solid #333;
                padding: 10px;
            }
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(10)
        
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("💭 Type a message...")
        self.input_line.returnPressed.connect(self.on_send)
        
        self.send_btn = QPushButton("📤 Send")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(self.on_send)
        self.send_btn.setFixedWidth(100)

        input_layout.addWidget(self.input_line)
        input_layout.addWidget(self.send_btn)
        chat_layout.addWidget(input_container)

        content_splitter.addWidget(chat_widget)

        # --- Right Panel (Config) ---
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                border-left: 2px solid #333;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(12)

        config_header = QLabel("⚙️ Configuration")
        config_header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                padding: 8px;
                background-color: #252525;
                border-radius: 6px;
            }
        """)
        right_layout.addWidget(config_header)

        # Document Selection
        doc_label = QLabel("📄 Document:")
        doc_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        right_layout.addWidget(doc_label)
        self.doc_combo = QComboBox()
        self.doc_combo.currentIndexChanged.connect(self.on_doc_changed)
        right_layout.addWidget(self.doc_combo)
        
        upload_btn = QPushButton("📁 Upload Document")
        upload_btn.clicked.connect(self.on_upload)
        right_layout.addWidget(upload_btn)

        right_layout.addSpacing(10)
        
        # LLM Selection
        llm_provider_label = QLabel("🧠 LLM Provider:")
        llm_provider_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        right_layout.addWidget(llm_provider_label)
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(["local", "ollama", "openai", "google"])
        self.llm_provider_combo.currentTextChanged.connect(self.refresh_llm_models)
        right_layout.addWidget(self.llm_provider_combo)

        llm_model_label = QLabel("LLM Model:")
        llm_model_label.setStyleSheet("color: #aaaaaa;")
        right_layout.addWidget(llm_model_label)
        self.llm_model_combo = QComboBox()
        right_layout.addWidget(self.llm_model_combo)

        right_layout.addSpacing(10)

        # Embedding Selection
        embed_provider_label = QLabel("🔤 Embedding Provider:")
        embed_provider_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        right_layout.addWidget(embed_provider_label)
        self.embed_provider_combo = QComboBox()
        self.embed_provider_combo.addItems(["local", "ollama", "openai", "google"])
        self.embed_provider_combo.currentTextChanged.connect(self.refresh_embed_models)
        right_layout.addWidget(self.embed_provider_combo)

        embed_model_label = QLabel("Embedding Model:")
        embed_model_label.setStyleSheet("color: #aaaaaa;")
        right_layout.addWidget(embed_model_label)
        self.embed_model_combo = QComboBox()
        right_layout.addWidget(self.embed_model_combo)

        right_layout.addSpacing(10)
        
        # Similarity Method Selection
        similarity_label = QLabel("🔍 Similarity Method:")
        similarity_label.setStyleSheet("font-weight: bold; color: #cccccc;")
        right_layout.addWidget(similarity_label)
        self.similarity_combo = QComboBox()
        self.similarity_combo.addItems(["cosine", "euclidean", "dot_product", "manhattan"])
        self.similarity_combo.setCurrentText(self.config.get("similarity_method", "cosine"))
        self.similarity_combo.currentTextChanged.connect(self.on_similarity_changed)
        right_layout.addWidget(self.similarity_combo)
        
        # Info label about similarity methods
        info_label = QLabel("• Cosine: Best for normalized vectors\n• Euclidean: Distance-based\n• Dot Product: Raw similarity\n• Manhattan: L1 distance")
        info_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 11px;
                padding: 8px;
                background-color: #252525;
                border-radius: 6px;
                border: 1px solid #333;
            }
        """)
        info_label.setWordWrap(True)
        right_layout.addWidget(info_label)

        right_layout.addStretch()
        
        # Status Bar
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        right_layout.addWidget(self.status_label)

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([800, 300])
        main_layout.addWidget(content_splitter)

        # Internal State
        self.current_chat_id = None
        self.current_doc_id = None
        self.chat_engine = None
        self.current_bot_bubble = None

        # Initialize
        self.refresh_chat_history()
        self.refresh_docs()
        self.refresh_llm_models()
        self.refresh_embed_models()
        
        # Load last config
        self.load_ui_state()

    # ---------------------------
    # UI Logic
    # ---------------------------
    def load_ui_state(self):
        # Restore last selected providers
        llm_prov = self.config.get("last_selected_llm_provider", "local")
        self.llm_provider_combo.setCurrentText(llm_prov)
        
        embed_prov = self.config.get("last_selected_embed_provider", "local")
        self.embed_provider_combo.setCurrentText(embed_prov)
        
        # Trigger refresh to populate models, then try to set model
        self.refresh_llm_models()
        self.refresh_embed_models()
        
        llm_mod = self.config.get("last_selected_llm_model", "")
        if llm_mod:
            self.llm_model_combo.setCurrentText(llm_mod)
            
        embed_mod = self.config.get("last_selected_embed_model", "")
        if embed_mod:
            self.embed_model_combo.setCurrentText(embed_mod)
        
        similarity_method = self.config.get("similarity_method", "cosine")
        if similarity_method:
            self.similarity_combo.setCurrentText(similarity_method)

    def save_ui_state(self):
        self.config.set("last_selected_llm_provider", self.llm_provider_combo.currentText())
        self.config.set("last_selected_llm_model", self.llm_model_combo.currentText())
        self.config.set("last_selected_embed_provider", self.embed_provider_combo.currentText())
        self.config.set("last_selected_embed_model", self.embed_model_combo.currentText())
        self.config.set("similarity_method", self.similarity_combo.currentText())
    
    def on_similarity_changed(self):
        self.config.set("similarity_method", self.similarity_combo.currentText())

    def refresh_chat_history(self):
        self.history_list.clear()
        chats = self.history_manager.list_chats()
        for chat in chats:
            item_text = chat["name"]
            # Store ID in user role or just use parallel list. 
            # QListWidget items can store data.
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, chat["id"])
            self.history_list.addItem(item)

    def refresh_docs(self):
        folder = "data/embeddings"
        os.makedirs(folder, exist_ok=True)
        self.doc_combo.clear()
        self.doc_combo.addItem("None", None)
        for f in os.listdir(folder):
            if f.endswith(".pkl"):
                doc_id = f[:-4]
                self.doc_combo.addItem(doc_id, doc_id)

    def refresh_llm_models(self):
        provider = self.llm_provider_combo.currentText()
        self.llm_model_combo.clear()
        
        models = []
        if provider == "local":
            models = list_local_llm_models() + list_custom_models()
        elif provider == "ollama":
            models = list_ollama_models()
        elif provider == "openai":
            models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
        elif provider == "google":
            models = ["gemini-pro", "gemini-1.5-flash"]
            
        self.llm_model_combo.addItems(models)

    def refresh_embed_models(self):
        provider = self.embed_provider_combo.currentText()
        self.embed_model_combo.clear()
        
        models = []
        if provider == "local":
            models = list_local_embedding_models() + list_custom_models()
        elif provider == "ollama":
            models = list_ollama_models()
        elif provider == "openai":
            models = ["text-embedding-3-small", "text-embedding-3-large"]
        elif provider == "google":
            models = ["models/embedding-001"]
            
        self.embed_model_combo.addItems(models)

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            # Refresh models in case paths changed
            self.refresh_llm_models()
            self.refresh_embed_models()

    def on_new_chat(self):
        self.current_chat_id = None
        self.chat_header.setText("New Chat")
        self.clear_chat_area()
        self.chat_engine = None
        # Deselect history list
        self.history_list.clearSelection()

    def on_history_select(self, item):
        chat_id = item.data(Qt.UserRole)
        self.load_chat(chat_id)

    def load_chat(self, chat_id):
        chat_data = self.history_manager.load_chat(chat_id)
        if not chat_data:
            return
        
        self.current_chat_id = chat_id
        self.chat_header.setText(chat_data.get("name", "Untitled"))
        
        # Set doc if associated
        doc_id = chat_data.get("doc_id")
        if doc_id:
            idx = self.doc_combo.findData(doc_id)
            if idx >= 0:
                self.doc_combo.setCurrentIndex(idx)
        
        # Rebuild UI
        self.clear_chat_area()
        for msg in chat_data.get("messages", []):
            role = msg.get("role")
            text = msg.get("content") or msg.get("text") # Handle both formats if any
            is_user = (role == "user")
            bubble = ChatBubble(text, is_user=is_user)
            self.ui.append_bubble.emit(bubble)
            
        # Initialize engine if doc is selected
        if doc_id:
            self.initialize_engine(doc_id)

    def clear_chat_area(self):
        # Remove all widgets from chat layout except the stretch at the end
        # Actually easier to just remove all and re-add stretch
        while self.chat_container_layout.count():
            item = self.chat_container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.chat_container_layout.addStretch()

    def on_doc_changed(self, index):
        doc_id = self.doc_combo.currentData()
        if doc_id:
            self.current_doc_id = doc_id
            self.initialize_engine(doc_id)
        else:
            self.current_doc_id = None
            self.chat_engine = None

    def initialize_engine(self, doc_id):
        # Get current model settings
        llm_prov = self.llm_provider_combo.currentText()
        llm_mod = self.llm_model_combo.currentText()
        embed_prov = self.embed_provider_combo.currentText()
        embed_mod = self.embed_model_combo.currentText()
        
        if not (llm_mod and embed_mod):
            return

        try:
            vs = VectorStore(doc_id)
            vs.load()
            
            embedder = Embedder(embed_prov, embed_mod)
            self.chat_engine = ChatEngine(
                vector_store=vs,
                embedder=embedder,
                provider=llm_prov,
                model_name=llm_mod,
                chat_id=self.current_chat_id
            )
            self.save_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize engine: {e}")

    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Document", "", "Documents (*.txt *.pdf *.docx)")
        if not path:
            return

        doc_id = sanitize_doc_id(path)
        vs = VectorStore(doc_id)

        if vs.load():
            QMessageBox.information(self, "Loaded", "Loaded existing embeddings.")
            self.refresh_docs()
            idx = self.doc_combo.findData(doc_id)
            if idx >= 0:
                self.doc_combo.setCurrentIndex(idx)
            return

        embed_prov = self.embed_provider_combo.currentText()
        embed_mod = self.embed_model_combo.currentText()
        
        if not embed_mod:
             QMessageBox.warning(self, "Error", "Please select an embedding model first.")
             return

        self.ui.set_status.emit("Embedding...")
        worker = EmbedWorker(path, doc_id, self.ui, embed_prov, embed_mod)
        worker.start()
        
        # Watcher to refresh when done
        threading.Thread(target=self._wait_for_embedding, args=(doc_id,), daemon=True).start()

    def _wait_for_embedding(self, doc_id):
        vs = VectorStore(doc_id)
        while not vs.exists():
            threading.Event().wait(0.5)
        
        self.ui.clear_status.emit()
        self.ui.run_in_main.emit(lambda: self.refresh_docs())
        # Select the new doc
        def select_new_doc():
            idx = self.doc_combo.findData(doc_id)
            if idx >= 0:
                self.doc_combo.setCurrentIndex(idx)
        self.ui.run_in_main.emit(select_new_doc)

    def on_send(self):
        query = self.input_line.text().strip()
        if not query:
            return

        if not self.chat_engine:
            # If no chat engine, check if we have a doc selected to init one
            if self.current_doc_id:
                self.initialize_engine(self.current_doc_id)
            
            if not self.chat_engine:
                 QMessageBox.warning(self, "Error", "Please select a document and configure models first.")
                 return

        # Create chat session if not exists
        if not self.current_chat_id:
            # Generate a name based on first query
            name = query[:30] + "..." if len(query) > 30 else query
            chat_data = self.history_manager.create_new_chat(name, self.current_doc_id)
            self.current_chat_id = chat_data["id"]
            self.chat_engine.chat_id = self.current_chat_id
            self.chat_header.setText(name)
            self.ui.refresh_history.emit()

        # UI Update
        user_bubble = ChatBubble(query, is_user=True)
        self.ui.append_bubble.emit(user_bubble)
        self.input_line.clear()
        self.ui.enable_input.emit(False)

        # Show loading indicator immediately
        def make_loading_bubble():
            self.current_bot_bubble = ChatBubble("", is_user=False, show_loading=True)
            self.ui.append_bubble.emit(self.current_bot_bubble)
        self.ui.run_in_main.emit(make_loading_bubble)

        # Save user message
        current_history = self.chat_engine.load_history()
        current_history.append({"role": "user", "content": query})
        self.chat_engine.save_history(current_history)

        # Stream Reply
        similarity_method = self.similarity_combo.currentText()
        threading.Thread(target=self._stream_reply, args=(query, similarity_method), daemon=True).start()

    def _stream_reply(self, query, similarity_method):
        try:
            print(f"Starting stream reply for query: {query}")
            retrieved_context, context_results = self.chat_engine.retrieve(query, similarity_method)
            print(f"Retrieved context, starting chat...")
            
            # Show context in collapsible dropdown (after loading, before answer)
            if context_results:
                def show_context():
                    context_bubble = ContextBubble(context_results)
                    self.ui.append_bubble.emit(context_bubble)
                self.ui.run_in_main.emit(show_context)
            
            stream = self.chat_engine.chat(query, retrieved_context, stream=True)
            print(f"Chat stream created, type: {type(stream)}")

            # Replace loading bubble with answer bubble BEFORE streaming starts
            def ensure_answer_bubble():
                # If we have a loading bubble, replace it with an answer bubble
                if self.current_bot_bubble and hasattr(self.current_bot_bubble, 'show_loading') and self.current_bot_bubble.show_loading:
                    # Find and remove the loading bubble
                    idx = self.chat_container_layout.indexOf(self.current_bot_bubble)
                    if idx >= 0:
                        item = self.chat_container_layout.takeAt(idx)
                        if item:
                            widget = item.widget()
                            if widget:
                                widget.deleteLater()
                    self.current_bot_bubble = None
                
                # Create new answer bubble and add it directly to the chat container
                if not self.current_bot_bubble:
                    self.current_bot_bubble = ChatBubble("", is_user=False)
                    # Ensure proper parent
                    self.current_bot_bubble.setParent(self.chat_container)
                    # Make sure it's visible
                    self.current_bot_bubble.show()
                    if hasattr(self.current_bot_bubble, 'label'):
                        self.current_bot_bubble.label.show()
                    # Add directly to layout to ensure it's in the main window
                    self.chat_container_layout.insertWidget(self.chat_container_layout.count() - 1, self.current_bot_bubble)
                    # Ensure container is visible
                    self.chat_container.show()
                    self.scroll.show()
                    self._scroll_to_bottom()
                    print("Answer bubble created and added DIRECTLY to chat container in main window")
            
            self.ui.run_in_main.emit(ensure_answer_bubble)
            
            # Small delay to ensure bubble is ready
            import time
            time.sleep(0.2)

            answer = ""
            packet_count = 0
            for packet in stream:
                packet_count += 1
                if packet_count <= 3:
                    print(f"DEBUG: Packet {packet_count}: {packet}")
                    
                content = packet.get("message", {}).get("content", "")
                if not content:
                    continue
                answer += content
                # Emit typing update to append to the answer bubble
                self.ui.typing_update.emit(content)
            
            print(f"Stream completed. Total packets: {packet_count}, Answer length: {len(answer)}")
            print(f"Answer content: {answer[:200]}...")  # Debug: print first 200 chars
            
            if not answer:
                print("WARNING: No answer generated!")
                # Replace loading bubble with error message
                def show_error():
                    if self.current_bot_bubble and hasattr(self.current_bot_bubble, 'show_loading') and self.current_bot_bubble.show_loading:
                        # Remove loading bubble
                        idx = self.chat_container_layout.indexOf(self.current_bot_bubble)
                        if idx >= 0:
                            item = self.chat_container_layout.takeAt(idx)
                            if item:
                                widget = item.widget()
                                if widget:
                                    widget.deleteLater()
                    error_bubble = ChatBubble("[No response generated. Please check the terminal for errors.]", is_user=False)
                    self.ui.append_bubble.emit(error_bubble)
                self.ui.run_in_main.emit(show_error)
            else:
                # Ensure answer is fully displayed (in case streaming didn't work or was incomplete)
                def ensure_answer_displayed():
                    if self.current_bot_bubble:
                        # Make sure bubble is visible
                        self.current_bot_bubble.show()
                        # If label is empty or has less text than answer, update it
                        current_text = self.current_bot_bubble.label.text() if hasattr(self.current_bot_bubble, 'label') else ""
                        if len(current_text) < len(answer):
                            if hasattr(self.current_bot_bubble, 'label'):
                                self.current_bot_bubble.label.setText(answer)
                                self.current_bot_bubble.label.show()
                            # Hide loading indicator if still visible
                            if hasattr(self.current_bot_bubble, 'show_loading') and self.current_bot_bubble.show_loading:
                                if hasattr(self.current_bot_bubble, 'loading_indicator') and self.current_bot_bubble.loading_indicator:
                                    self.current_bot_bubble.loading_indicator.hide()
                                self.current_bot_bubble.show_loading = False
                        print(f"Answer displayed in UI. Length: {len(answer)}, Bubble visible: {self.current_bot_bubble.isVisible()}")
                    else:
                        print("ERROR: No answer bubble exists to display answer!")
                        # Create answer bubble as fallback
                        self.current_bot_bubble = ChatBubble(answer, is_user=False)
                        self.current_bot_bubble.show()
                        self.ui.append_bubble.emit(self.current_bot_bubble)
                self.ui.run_in_main.emit(ensure_answer_displayed)

            # Save assistant message
            current_history = self.chat_engine.load_history()
            current_history.append({"role": "assistant", "content": answer})
            self.chat_engine.save_history(current_history)
            
            # Update history list (timestamp changed)
            self.ui.refresh_history.emit()

        except Exception as e:
            msg = str(e)
            print(f"ERROR in _stream_reply: {msg}")
            import traceback
            traceback.print_exc()
            def show_error():
                if self.current_bot_bubble and hasattr(self.current_bot_bubble, 'show_loading') and self.current_bot_bubble.show_loading:
                    # Remove loading bubble
                    idx = self.chat_container_layout.indexOf(self.current_bot_bubble)
                    if idx >= 0:
                        item = self.chat_container_layout.takeAt(idx)
                        if item:
                            widget = item.widget()
                            if widget:
                                widget.deleteLater()
                error_bubble = ChatBubble(f"[Error] {msg}", is_user=False)
                self.ui.append_bubble.emit(error_bubble)
            self.ui.run_in_main.emit(show_error)
        finally:
            self.ui.enable_input.emit(True)


    # ---------------------------
    # Slot Helpers
    # ---------------------------
    def _slot_append_bubble(self, bubble):
        # Ensure bubble is visible before adding
        bubble.show()
        bubble.setParent(self.chat_container)  # Ensure proper parent
        if hasattr(bubble, 'label'):
            bubble.label.show()
        # Insert before the stretch at the end
        self.chat_container_layout.insertWidget(self.chat_container_layout.count() - 1, bubble)
        # Ensure container and scroll area are visible
        self.chat_container.show()
        self.scroll.show()
        self._scroll_to_bottom()
        # Force update and repaint
        QApplication.processEvents()
        bubble.update()
        self.chat_container.update()

    def _slot_typing_update(self, text):
        if self.current_bot_bubble:
            # Make sure the bubble is ready to receive text
            if hasattr(self.current_bot_bubble, 'show_loading') and self.current_bot_bubble.show_loading:
                # If still showing loading, hide loading indicator and show label
                if hasattr(self.current_bot_bubble, 'loading_indicator') and self.current_bot_bubble.loading_indicator:
                    self.current_bot_bubble.loading_indicator.hide()
                    self.current_bot_bubble.loading_indicator = None
                self.current_bot_bubble.show_loading = False
                if hasattr(self.current_bot_bubble, 'label'):
                    self.current_bot_bubble.label.show()
            
            # Ensure bubble and label are visible
            self.current_bot_bubble.show()
            if hasattr(self.current_bot_bubble, 'label'):
                self.current_bot_bubble.label.show()
            
            self.current_bot_bubble.append_text(text)
            self._scroll_to_bottom()
        else:
            # If no bubble exists, create one immediately
            print("WARNING: No answer bubble exists, creating one now")
            self.current_bot_bubble = ChatBubble(text, is_user=False)
            self.current_bot_bubble.show()
            self.current_bot_bubble.label.show()
            self.chat_container_layout.insertWidget(self.chat_container_layout.count() - 1, self.current_bot_bubble)
            self._scroll_to_bottom()

    def _slot_set_status(self, msg):
        self.status_label.setText(msg)

    def _slot_clear_status(self):
        self.status_label.setText("")

    def _slot_enable_input(self, enable):
        self.input_line.setEnabled(enable)
        self.send_btn.setEnabled(enable)

    def _slot_run_in_main(self, fn):
        try:
            fn()
        except Exception as e:
            print("Exception in run_in_main:", e)

    def _scroll_to_bottom(self):
        # Process events to ensure layout update before scrolling
        QApplication.processEvents()
        # Ensure scroll area is visible and updated
        self.scroll.show()
        self.chat_container.show()
        self.chat_container.update()
        vsb = self.scroll.verticalScrollBar()
        if vsb:
            vsb.setValue(vsb.maximum())
            QApplication.processEvents()  # Process again after scrolling
