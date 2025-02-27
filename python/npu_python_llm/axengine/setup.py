# Copyright (c) 2024, Axera Tech. All rights reserved.

"""
Setup AXEngine Python Package.
"""

import os
import subprocess

from pathlib import Path

from setuptools import find_packages, setup

SOURCE_DIR = Path(__file__).parent.resolve()

PACKAGE_VERSION = "0.2.0a0"
GIT_HASH = "Unknown"
PACKAGE_NAME = "axengine"

try:
    GIT_HASH = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_DIR).decode("ascii").strip()
except Exception:
    pass

if os.getenv("BUILD_VERSION"):
    PACKAGE_VERSION = os.getenv("BUILD_VERSION")
elif GIT_HASH != "Unknown":
    PACKAGE_VERSION += "+" + GIT_HASH[:8]


def write_version_file():
    version_path = SOURCE_DIR / PACKAGE_NAME / "version.py"
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{PACKAGE_VERSION}'\n")
        f.write(f"git_version = {repr(GIT_HASH)}\n")


def load_requirements(path_dir=SOURCE_DIR, file_name="requirements.txt", comment_char="#"):
    with open(path_dir / file_name, "r", encoding="utf-8", errors="ignore") as file:
        lines = [ln.rstrip() for ln in file.readlines() if not ln.startswith("#")]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[: ln.index(comment_char)].strip()
        if ln.startswith("http"):  # skip directly installed dependencies
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


if __name__ == "__main__":
    print(f"Building wheel {PACKAGE_NAME}-{PACKAGE_VERSION}")

    write_version_file()

    setup(
        name=PACKAGE_NAME,
        version=PACKAGE_VERSION,
        description="AXera Engine",
        long_description="AXera Engine",
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=["test"]),
        include_package_data=True,
        install_requires=load_requirements(),
        zip_safe=False,
    )
