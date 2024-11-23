import pathlib

import fast_feature

PACKAGE_ROOT = pathlib.Path(fast_feature.__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "VERSION"

name = "fast_feature"

with open(VERSION_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
