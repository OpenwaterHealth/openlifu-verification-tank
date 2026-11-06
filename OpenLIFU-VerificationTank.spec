# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for the OpenLIFU verification tank report GUI.
#
# Produces one folder (dist/VerificationTank/) with two executables that
# share the same bundled libraries:
#   OpenLIFU Verification Tank.exe            - GUI, no console window
#   OpenLIFU Verification Tank (console).exe  - same app with a console
#                                               (log output)
# Both use the app icon (openlifu_verification/assets/app_icon.ico; edit
# app_icon.svg and run freeze/make_icon.py to regenerate it).
#
# Build:  python -m PyInstaller -y OpenLIFU-VerificationTank.spec
#
# The PicoSDK driver (ps5000a.dll) is NOT bundled; it must be installed on
# the target machine, as for a source install.
import os

from PyInstaller.utils.hooks import collect_all, copy_metadata

APP_NAME = "OpenLIFU Verification Tank"
# Output folder under dist/; kept so the install layout doesn't change.
DIST_DIR = "VerificationTank"
ROOT = SPECPATH
ICON = os.path.join(ROOT, "openlifu_verification", "assets", "app_icon.ico")

datas = [
    # Setup data template (package data, located via paths.py).
    (os.path.join(ROOT, "openlifu_verification", "setup_data_template"),
     os.path.join("openlifu_verification", "setup_data_template")),
    # Window / taskbar icon (located via paths.APP_ICON).
    (os.path.join(ROOT, "openlifu_verification", "assets"),
     os.path.join("openlifu_verification", "assets")),
]
# The app version shown in the GUI / reports comes from package metadata.
datas += copy_metadata("openlifu-verification")
binaries = []
hiddenimports = []

# openlifu_sdk ships libusb + data files; serial / usb are loaded by it.
for pkg in ("openlifu_sdk", "serial", "usb"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    [os.path.join(ROOT, "scripts", "run_report_gui.py")],
    # scripts/ for the sibling ``import run_report``; ROOT so the package
    # is analysed from source even with an editable install.
    pathex=[ROOT, os.path.join(ROOT, "scripts")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(ROOT, "freeze", "rth_null_stdio.py")],
    # Other Qt bindings, plus optional pandas / matplotlib / xarray
    # integrations the app never uses (pulled in when present in the
    # build environment; netCDF4 alone adds ~50 MB). xarray itself is
    # required by openlifu_sdk.
    excludes=[
        "PyQt6", "PyQt5", "PySide2",
        "netCDF4", "cftime", "h5py", "zarr", "numcodecs",
        "IPython", "ipykernel", "jedi", "notebook", "torch",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    icon=ICON,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

exe_console = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=f"{APP_NAME} (console)",
    icon=ICON,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe_gui,
    exe_console,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=DIST_DIR,
)
