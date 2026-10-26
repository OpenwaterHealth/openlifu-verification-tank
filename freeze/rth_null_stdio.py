# PyInstaller runtime hook: the windowed (console=False) exe starts with
# sys.stdout / sys.stderr set to None, and tqdm progress bars used by the
# scans crash when writing to them. Point them at os.devnull instead.
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
