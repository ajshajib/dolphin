.. |logo| image:: https://raw.githubusercontent.com/ajshajib/dolphin/efb2673646edd6c2d98963e9f4d08a9104d293c3/logo.png
    :width: 70

|logo| dolphin
==============

.. image:: https://readthedocs.org/projects/dolphin-docs/badge/?version=latest
    :target: https://dolphin-docs.readthedocs.io/latest/
.. image:: https://github.com/ajshajib/dolphin/actions/workflows/ci.yaml/badge.svg?branch=main
    :target: https://github.com/ajshajib/dolphin/actions/workflows/ci.yaml
.. image:: https://codecov.io/gh/ajshajib/dolphin/branch/main/graph/badge.svg?token=WZVXZS9GF1 
    :target: https://app.codecov.io/gh/ajshajib/dolphin/tree/main
    :alt: Codecov
.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/ajshajib/dolphin/blob/main/LICENSE
    :alt: License BSD 3-Clause Badge
.. image:: https://img.shields.io/badge/ApJ-%20992%2040-D22630
   :target: https://iopscience.iop.org/article/10.3847/1538-4357/adf95c
   :alt: Shajib et al. 2025, ApJ, 992, 40
.. image:: https://img.shields.io/badge/arXiv-2503.22657-b31b1b?logo=arxiv&logoColor=white
    :target: https://arxiv.org/abs/2503.22657
.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16587211-blue
   :target: https://doi.org/10.5281/zenodo.16587211
   :alt: Zenodo DOI 10.5281/zenodo.16587211
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=brightyellow
    :target: https://pre-commit.com/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg
    :target: https://github.com/PyCQA/docformatter
.. image:: https://img.shields.io/badge/%20style-sphinx-0a507a.svg
    :target: https://www.sphinx-doc.org/en/master/usage/index.html

Welcome to **dolphin**, an AI-powered automated pipeline for strong gravitational lens modeling! 

``dolphin`` leverages `lenstronomy <https://github.com/lenstronomy/lenstronomy>`_ as its core modeling engine, providing an accessible and scalable framework for studying galaxy-scale lenses.

What is Dolphin?
----------------

Strong gravitational lens modeling traditionally requires significant manual effort. ``dolphin`` changes this by providing an AI-driven approach to forward modeling, enabling researchers to process large samples of strong lenses with ease. Whether you want a fully hands-off automated pipeline or a semi-automated workflow where you can fine-tune the AI-generated configurations, ``dolphin`` gives you the flexibility and power you need.

Features
--------

- 🤖 **AI-Automated Modeling**: Streamline forward modeling for large datasets of galaxy-scale lenses.
- 🎛️ **Flexible Workflows**: Choose between fully automated runs or semi-automated modes with manual overrides.
- 🌈 **Multi-Band Support**: Easily configure and model across multiple observing bands simultaneously.
- 🌌 **Versatile Sources**: Built-in support for both **galaxy–galaxy** and **galaxy–quasar** lens systems.
- 💻 **HPC Ready**: Seamlessly sync your setup between local machines and High-Performance Computing Clusters (HPCC).
- ✅ **Tested & Reliable**: Comprehensively tested with |Codecov|.

.. |Codecov| image:: https://codecov.io/gh/ajshajib/dolphin/branch/main/graph/badge.svg?token=WZVXZS9GF1 
    :target: https://app.codecov.io/gh/ajshajib/dolphin/tree/main

Installation
------------

.. image:: https://img.shields.io/pypi/v/space-dolphin.svg
   :alt: PyPI - Version
   :target: https://pypi.org/project/space-dolphin/

Installing ``dolphin`` is simple. You can install the latest stable release via ``pip``:

.. code-block:: bash

    pip install space-dolphin

Alternatively, to install the latest development version directly from GitHub:

.. code-block:: bash

    git clone https://github.com/ajshajib/dolphin.git
    cd dolphin
    pip install .

For instructions on setting up your workspace and running your first model, please check out our `Quickstart guide <QUICKSTART.rst>`_.

Citation
--------

If you use ``dolphin`` in your research, please cite the main ``dolphin`` paper:

- `Shajib et al. (2025) <https://arxiv.org/abs/2503.22657>`_

Depending on the fitting recipe used, please additionally cite the following papers for the underlying modeling methodology:

- **Galaxy-Quasar Recipe:** `Shajib et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.5649S/abstract>`_
- **Galaxy-Galaxy Recipe:** `Shajib et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.2380S/abstract>`_
