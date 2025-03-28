#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

test_requirements = []

setup(
    author="Anowar Shajib",
    author_email="ajshajib@gmail.com",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Users",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    description="Automated pipeline for lens modeling based on lenstronomy",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="dolphin",
    name="dolphin",
    packages=find_packages(include=["dolphin", "dolphin.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ajshajib/dolphin",
    version="1.0.0",
    zip_safe=False,
)
