"""The app icon ships with the package and has every Windows icon size."""
from __future__ import annotations

from PIL import Image

from openlifu_verification.paths import APP_ICON


def test_app_icon_has_all_sizes():
    assert APP_ICON.is_file()
    with Image.open(APP_ICON) as ico:
        assert set(ico.info["sizes"]) >= {(s, s) for s in (16, 24, 32, 48, 256)}


def test_app_icon_sources_are_alongside():
    assert APP_ICON.with_suffix(".svg").is_file()
    with Image.open(APP_ICON.with_suffix(".png")) as png:
        assert png.size == (256, 256)
