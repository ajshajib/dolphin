#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
readme = (this_directory / "README.rst").read_text()

requirements = []

test_requirements = []

setup(
    author="Anowar J. Shajib",
    author_email="ajshajib@gmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Users",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="Automated pipeline for lens modeling based on lenstronomy",
    install_requires=requirements,
    license="BSD 3-Clause License",
    long_description=readme,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="dolphin",
    name="dolphin",
    packages=find_packages(include=["dolphin", "dolphin.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ajshajib/dolphin",
    version="1.0.2",
    zip_safe=False,
)
