# <img src="https://raw.githubusercontent.com/ajshajib/dolphin/efb2673646edd6c2d98963e9f4d08a9104d293c3/logo.png" alt="logo" width="40"/> dolphin

[![GitHub](https://github.com/ajshajib/dolphin/workflows/CI/badge.svg)](https://github.com/ajshajib/dolphin/actions)
[![docs](https://readthedocs.org/projects/dolphin-docs/badge/?version=latest)](https://dolphin-docs.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ajshajib/dolphin/graph/badge.svg?token=WZVXZS9GF1)](https://codecov.io/gh/ajshajib/dolphin)
[![pypi](https://img.shields.io/pypi/v/space-dolphin)](https://pypi.org/project/space-dolphin)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![sphinx](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)

AI-powered automated pipeline for lens modeling, with
[`lenstronomy`](https://github.com/sibirrer/lenstronomy) as the modeling engine.

## Features
- **AI-powered automated** modeling for large samples of galaxy-scale lenses.
- **Flexible**: fully automated or semi-automated with user tweaks.
- **Multi-band** lens modeling made simple. 
- Supports **both** galaxy–galaxy and galaxy–quasar systems.
- Effortless syncing between local machines and **HPCC**. 
- [![codecov](https://codecov.io/gh/ajshajib/dolphin/graph/badge.svg?token=WZVXZS9GF1)](https://codecov.io/gh/ajshajib/dolphin) **tested!**

## Installation

You can install `dolphin` using `pip`. Run the following command:

```bash
pip install space-dolphin
```

Alternatively, you can install the latest development version from GitHub as:

```bash
git clone https://github.com/ajshajib/dolphin.git
cd dolphin
pip install .
```

See the [Quickstart guide](QUICKSTART.rst) for instructions on setting up and running `dolphin`.

## Citation

If you use `dolphin` in your research, please cite the `dolphin` paper [Shajib et al. (2025)](). If you have used the `"galaxy-quasar"` fitting recipe, then additionally cite [Shajib et al. (2019)
](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.5649S/abstract), or if you have used the `"galaxy-galaxy"` fitting recipe, then cite [Shajib et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.2380S/abstract).
