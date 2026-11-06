"""Render the app icon SVG to the .ico / .png files the app and build use.

Run after editing ``openlifu_verification/assets/app_icon.svg``:

    python freeze/make_icon.py

Writes ``app_icon.ico`` (16-256 px, used for the exe and the window /
taskbar icon) and ``app_icon.png`` (256 px) next to the SVG. Each size
is rendered from the vector source rather than downscaled.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QBuffer, QIODevice, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "openlifu_verification" / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def render(renderer: QSvgRenderer, size: int) -> Image.Image:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def main() -> int:
    _app = QGuiApplication(sys.argv[:1])
    renderer = QSvgRenderer(str(ASSETS / "app_icon.svg"))
    if not renderer.isValid():
        raise SystemExit("could not parse app_icon.svg")
    images = [render(renderer, s) for s in SIZES]
    images[-1].save(ASSETS / "app_icon.png")
    images[-1].save(ASSETS / "app_icon.ico", sizes=[(s, s) for s in SIZES],
                    append_images=images[:-1])
    print(f"wrote {ASSETS / 'app_icon.ico'} and app_icon.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
