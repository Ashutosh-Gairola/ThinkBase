DARK_THEME = """
/* Global Reset */
QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    font-size: 14px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #2d2d2d;
    width: 12px;
    margin: 0px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: linear-gradient(to bottom, #4a4a4a, #5a5a5a);
    min-height: 30px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: linear-gradient(to bottom, #5a5a5a, #6a6a6a);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #2d2d2d;
    height: 12px;
    margin: 0px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background: linear-gradient(to right, #4a4a4a, #5a5a5a);
    min-width: 30px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background: linear-gradient(to right, #5a5a5a, #6a6a6a);
}

/* Buttons */
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #3a3a3a);
    border: 1px solid #5a5a5a;
    border-radius: 8px;
    padding: 10px 20px;
    color: #ffffff;
    font-weight: 500;
    min-height: 20px;
}
QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a5a5a, stop:1 #4a4a4a);
    border-color: #6a6a6a;
    transform: translateY(-1px);
}
QPushButton:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3a3a3a, stop:1 #2a2a2a);
    border-color: #4a4a4a;
}
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666666;
    border-color: #333333;
}

/* Primary Button */
QPushButton#primary {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0078d4, stop:1 #0063b1);
    border-color: #0078d4;
    box-shadow: 0px 2px 4px rgba(0, 120, 212, 0.3);
}
QPushButton#primary:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1084e0, stop:1 #0078d4);
    border-color: #1084e0;
    box-shadow: 0px 4px 8px rgba(0, 120, 212, 0.4);
}
QPushButton#primary:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0063b1, stop:1 #005a9e);
}

/* Inputs */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    padding: 10px 12px;
    color: #ffffff;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #0078d4;
    background-color: #252525;
    box-shadow: 0px 0px 0px 2px rgba(0, 120, 212, 0.2);
}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover {
    border-color: #4a4a4a;
}

/* Lists */
QListWidget {
    background-color: #252526;
    border: 2px solid #333333;
    border-radius: 8px;
    padding: 4px;
}
QListWidget::item {
    padding: 10px;
    border-radius: 6px;
    margin: 2px;
}
QListWidget::item:selected {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #37373d, stop:1 #2d2d2d);
    color: #ffffff;
    border: 1px solid #0078d4;
}
QListWidget::item:hover {
    background-color: #2a2d2e;
}

/* Combo Box */
QComboBox {
    background-color: #2d2d2d;
    border: 2px solid #3d3d3d;
    border-radius: 8px;
    padding: 8px 12px;
    color: #ffffff;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #4a4a4a;
}
QComboBox:focus {
    border-color: #0078d4;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
    border-left: 2px solid #3d3d3d;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}
QComboBox::drop-down:hover {
    background-color: #3a3a3a;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 2px solid #0078d4;
    border-radius: 8px;
    selection-background-color: #0078d4;
    selection-color: #ffffff;
    padding: 4px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #cccccc;
    width: 0px;
    height: 0px;
    margin-right: 8px;
}

/* Labels */
QLabel {
    color: #e0e0e0;
}
QLabel#header {
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
    padding: 10px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078d4, stop:1 #1084e0);
    border-radius: 8px;
    margin: 5px;
}

/* Chat Bubbles */
QFrame#user_bubble {
    background-color: #0078d4;
    border-radius: 12px;
    border-bottom-right-radius: 2px;
}
QFrame#bot_bubble {
    background-color: #333333;
    border-radius: 12px;
    border-bottom-left-radius: 2px;
}

/* Splitter */
QSplitter::handle {
    background-color: #333333;
    width: 2px;
}
QSplitter::handle:hover {
    background-color: #0078d4;
}

/* Scroll Area */
QScrollArea {
    border: none;
    background-color: #1e1e1e;
}

/* Status Bar */
QStatusBar {
    background-color: #252526;
    border-top: 1px solid #333333;
}
"""
