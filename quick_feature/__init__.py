import pathlib

import quick_feature

PACKAGE_ROOT = pathlib.Path(quick_feature.__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "VERSION"

name = "quick_feature"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
