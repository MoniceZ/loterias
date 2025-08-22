import sys
from PyQt6.QtWidgets import QApplication
from ui import App

def main() -> None:
    app = QApplication(sys.argv)
    janela = App()
    janela.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
