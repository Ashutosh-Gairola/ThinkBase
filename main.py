import sys
import os
from PySide6.QtWidgets import QApplication

from ui.model_download_dialog import ModelDownloadDialog
from ui.app_window import MainWindow


def ensure_models():
    emb_folder = "offline_models/embedding"
    llm_folder = "offline_models/language"

    os.makedirs(emb_folder, exist_ok=True)
    os.makedirs(llm_folder, exist_ok=True)

    emb_missing = len(os.listdir(emb_folder)) == 0
    llm_missing = len(os.listdir(llm_folder)) == 0

    if emb_missing or llm_missing:
        dlg = ModelDownloadDialog()
        dlg.exec()


def main():
    app = QApplication(sys.argv)

    ensure_models()  # <-- popup FIRST

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
