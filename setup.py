#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

readme = open("README.rst").read()
doclink = """
Documentation
-------------

The full documentation is at http://dolphin.rtfd.org."""
history = open("HISTORY.rst").read().replace(".. :changelog:", "")

setup(
    name="dolphin",
    version="0.0.0",
    description="Automated pipeline for lens modeling based on lenstronomy.",
    long_description=readme + "\n\n" + doclink + "\n\n" + history,
    author="Anowar J. Shajib",
    author_email="ajshajib@gmail.com",
    url="https://github.com/ajshajib/dolphin",
    packages=[
        "dolphin",
    ],
    package_dir={"dolphin": "dolphin"},
    include_package_data=True,
    install_requires=[],
    license="MIT",
    zip_safe=False,
    keywords="dolphin",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        #'Programming Language :: Python :: 2.7',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        #'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
