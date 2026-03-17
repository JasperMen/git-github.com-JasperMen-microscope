from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from camera_window import CameraWindow


def main() -> None:
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(11)
    app.setFont(font)

    window = CameraWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
