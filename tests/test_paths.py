"""Tests for :mod:`openlifu_verification.paths`."""
from openlifu_verification import paths


def test_template_is_package_data():
    """The template must live inside the package so it ships with a
    regular (non-editable) install."""
    package_dir = paths.BUNDLED_SETUP_DATA_DIR.parent
    assert (package_dir / "__init__.py").is_file()
    for n in (1, 2):
        assert (paths.BUNDLED_SETUP_DATA_DIR / "transducers"
                / f"openlifu_{n}x.json").is_file()


def test_bundled_calibrations_dir_ships_a_file():
    assert paths.BUNDLED_CALIBRATIONS_DIR.is_dir()
    hits = list(paths.BUNDLED_CALIBRATIONS_DIR.glob("HNR0500-*.txt"))
    assert hits, f"no HNR0500 calibrations found in {paths.BUNDLED_CALIBRATIONS_DIR}"
