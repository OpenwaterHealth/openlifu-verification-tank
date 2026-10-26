# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for the OpenLIFU verification tank report GUI.
#
# Produces one folder (dist/VerificationTank/) with two executables that
# share the same bundled libraries:
#   VerificationTank.exe          - GUI, no console window
#   VerificationTank_console.exe  - same app with a console (log output)
#
# Build:  python -m PyInstaller -y OpenLIFU-VerificationTank.spec
#
# The PicoSDK driver (ps5000a.dll) is NOT bundled; it must be installed on
# the target machine, as for a source install.
import os

from PyInstaller.utils.hooks import collect_all, copy_metadata

APP_NAME = "VerificationTank"
ROOT = SPECPATH

datas = [
    # Resolved relative to the openlifu_verification package's parent
    # folder (paths.REPO_ROOT, VerificationTank transducer JSON), which is
    # the bundle root (_internal/) when frozen.
    (os.path.join(ROOT, "setup_data"), "setup_data"),
    (os.path.join(ROOT, "transducers"), "transducers"),
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
        "PySide6", "shiboken6", "PySide2", "PyQt5",
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
    name=f"{APP_NAME}_console",
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
    name=APP_NAME,
)
