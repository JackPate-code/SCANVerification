"""Launcher for the component inspection UI and PDF report pipeline.

Keep main.py, model.py, pdf.py, and UI.py in the same directory.
Run `python main.py` to open the UI. Use File > Generate sample inspection + PDF
to generate a model-backed report that appears in the UI's Report tab.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

def main() -> None:
    from PySide6.QtWidgets import QApplication
    import UI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = UI.MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
