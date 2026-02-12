from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame, QPushButton
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor

class LoadingIndicator(QWidget):
    """Animated loading dots indicator."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(50, 20)
        self.dots = [0.0, 0.0, 0.0]
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(150)
        
    def animate(self):
        # Animate dots with phase offset
        for i in range(3):
            self.dots[i] = (self.dots[i] + 0.2) % 1.0
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        dot_size = 8
        spacing = 12
        start_x = (self.width() - (dot_size * 3 + spacing * 2)) / 2
        
        for i in range(3):
            x = start_x + i * (dot_size + spacing)
            y = self.height() / 2
            
            # Opacity based on animation phase
            opacity = 0.3 + 0.7 * abs(0.5 - self.dots[i]) * 2
            color = QColor(150, 150, 150, int(255 * opacity))
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(int(x), int(y - dot_size/2), dot_size, dot_size)

class ContextBubble(QWidget):
    """Collapsible widget to display retrieved context chunks."""
    def __init__(self, contexts: list):
        super().__init__()
        self.contexts = contexts
        self.is_expanded = False
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(0)
        
        # Header button (clickable to expand/collapse)
        self.header_btn = QPushButton("📚 Retrieved Context (Click to expand)")
        self.header_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a3a2a;
                border: 1px solid #4CAF50;
                border-radius: 8px;
                padding: 8px 12px;
                color: #4CAF50;
                font-weight: bold;
                font-size: 12px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #2d4a2d;
                border-color: #66BB6A;
            }
        """)
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.clicked.connect(self.toggle_expand)
        main_layout.addWidget(self.header_btn)
        
        # Content area (initially hidden)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)
        self.content_widget.hide()
        
        # Add context items
        for i, (chunk, score) in enumerate(contexts, 1):
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #1a2a1a;
                    border: 1px solid #4CAF50;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(8, 8, 8, 8)
            
            score_label = QLabel(f"[{i}] Similarity Score: {score:.4f}")
            score_label.setStyleSheet("color: #81C784; font-size: 11px; font-weight: bold;")
            frame_layout.addWidget(score_label)
            
            chunk_label = QLabel(chunk[:500] + ("..." if len(chunk) > 500 else ""))
            chunk_label.setWordWrap(True)
            chunk_label.setStyleSheet("color: #e0e0e0; font-size: 12px; padding-top: 4px;")
            chunk_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            frame_layout.addWidget(chunk_label)
            
            self.content_layout.addWidget(frame)
        
        main_layout.addWidget(self.content_widget)
    
    def toggle_expand(self):
        """Toggle the expansion state of the context widget."""
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.content_widget.show()
            self.header_btn.setText("📚 Retrieved Context (Click to collapse)")
        else:
            self.content_widget.hide()
            self.header_btn.setText("📚 Retrieved Context (Click to expand)")

class ChatBubble(QWidget):
    """
    Simple chat bubble widget.
    - is_user=True => right aligned (green)
    - is_user=False => left aligned (dark)
    """


    def __init__(self, text: str = "", is_user: bool = False, show_loading: bool = False):
        super().__init__()
        self.is_user = is_user
        self.show_loading = show_loading

        # --- Thinking Section (Collapsible) ---
        self.thinking_widget = QWidget()
        self.thinking_layout = QVBoxLayout(self.thinking_widget)
        self.thinking_layout.setContentsMargins(0, 0, 0, 8)
        self.thinking_layout.setSpacing(4)
        
        self.thinking_header = QPushButton("🤔 Thinking Process (Click to expand)")
        self.thinking_header.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 6px 10px;
                color: #aaa;
                font-size: 11px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #444;
                color: #fff;
            }
        """)
        self.thinking_header.setCursor(Qt.PointingHandCursor)
        self.thinking_header.clicked.connect(self.toggle_thinking)
        
        self.thinking_content = QLabel("")
        self.thinking_content.setWordWrap(True)
        self.thinking_content.setStyleSheet("color: #888; font-size: 11px; padding: 4px; font-family: monospace;")
        self.thinking_content.hide()
        
        self.thinking_layout.addWidget(self.thinking_header)
        self.thinking_layout.addWidget(self.thinking_content)
        self.thinking_widget.hide() # Hidden by default until we have thoughts

        # --- Main Message Label ---
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # Hide label if showing loading
        if show_loading:
            self.label.hide()

        # Loading indicator
        self.loading_indicator = LoadingIndicator() if show_loading else None

        # style per side
        if is_user:
            # WhatsApp-style green bubble (right)
            self.label.setStyleSheet(
                """
                QLabel {
                    background-color: #25D366;
                    color: black;
                    padding: 10px 14px;
                    border-radius: 12px;
                    max-width: 480px;
                    font-size: 14px;
                }
                """
            )
        else:
            # Bot dark bubble (left)
            self.label.setStyleSheet(
                """
                QLabel {
                    background-color: #2f2f2f;
                    color: #f1f1f1;
                    padding: 10px 14px;
                    border-radius: 12px;
                    max-width: 520px;
                    font-size: 14px;
                }
                """
            )

        # Global container for bubble (Vertical: Thinking on top, Message below)
        self.container_layout = QVBoxLayout()
        self.container_layout.setContentsMargins(0,0,0,0)
        self.container_layout.setSpacing(0)
        
        # Add components to vertical container
        if not is_user:
            self.container_layout.addWidget(self.thinking_widget)
            
        # Message row (horizontal for alignment)
        msg_row = QHBoxLayout()
        msg_row.setContentsMargins(0,0,0,0)
        
        if is_user:
            msg_row.addStretch()
            msg_row.addWidget(self.label, 0, Qt.AlignRight)
        else:
            if show_loading and self.loading_indicator:
                # Create container for loading indicator
                container = QWidget()
                container_layout = QHBoxLayout(container)
                container_layout.setContentsMargins(10, 10, 10, 10)
                container_layout.addWidget(self.loading_indicator)
                container_layout.addStretch()
                container.setStyleSheet("""
                    QWidget {
                        background-color: #2f2f2f;
                        border-radius: 12px;
                        max-width: 520px;
                    }
                """)
                msg_row.addWidget(container, 0, Qt.AlignLeft)
            else:
                msg_row.addWidget(self.label, 0, Qt.AlignLeft)
            msg_row.addStretch()
            
        self.container_layout.addLayout(msg_row)
        
        # Main layout of this QWidget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addLayout(self.container_layout)
        self.setLayout(layout)

    def toggle_thinking(self):
        if self.thinking_content.isVisible():
            self.thinking_content.hide()
            self.thinking_header.setText("🤔 Thinking Process (Click to expand)")
        else:
            self.thinking_content.show()
            self.thinking_header.setText("🤔 Thinking Process (Click to collapse)")

    def append_thought(self, text: str):
        """Append to the thinking section."""
        if not self.thinking_widget.isVisible():
            self.thinking_widget.show()
        
        current = self.thinking_content.text()
        self.thinking_content.setText(current + text + "\n")

    def append_text(self, text: str):
        """Append streaming text to the bubble (main thread only)."""
        # Hide loading indicator when text starts coming
        if self.show_loading and self.loading_indicator:
            self.loading_indicator.hide()
            self.loading_indicator = None
            # Show the label instead
            self.label.show()
        
        current = self.label.text()
        self.label.setText(current + text)
