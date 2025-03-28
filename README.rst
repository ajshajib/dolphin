.. |logo| image:: https://raw.githubusercontent.com/ajshajib/dolphin/efb2673646edd6c2d98963e9f4d08a9104d293c3/logo.png
    :width: 70

|logo| dolphin
=======

.. image:: https://github.com/ajshajib/dolphin/workflows/Tests/badge.svg
    :target: https://github.com/ajshajib/dolphin/actions
.. image:: https://readthedocs.org/projects/dolphin-docs/badge/?version=latest
    :target: https://dolphin-docs.readthedocs.io/en/latest/?badge=latest
.. image:: https://codecov.io/gh/ajshajib/dolphin/graph/badge.svg?token=WZVXZS9GF1
    :target: https://codecov.io/gh/ajshajib/dolphin
.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
    :target: https://github.com/ajshajib/dolphin/blob/main/LICENSE
    :alt: License BSD 3-Clause Badge
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg
    :target: https://github.com/PyCQA/docformatter
.. image:: https://img.shields.io/badge/%20style-sphinx-0a507a.svg
    :target: https://www.sphinx-doc.org/en/master/usage/index.html

AI-powered automated pipeline for lens modeling, with
`lenstronomy <https://github.com/sibirrer/lenstronomy>`_ as the modeling engine.

Features
--------

- **AI-powered automated** modeling for large samples of galaxy-scale lenses.
- **Flexible**: fully automated or semi-automated with user tweaks.
- **Multi-band** lens modeling made simple.
- Supports **both** galaxy–galaxy and galaxy–quasar systems.
- Effortless syncing between local machines and **HPCC**.
- |codecov| **tested!**

.. |codecov| image:: https://codecov.io/gh/ajshajib/dolphin/graph/badge.svg?token=WZVXZS9GF1
      :target: https://codecov.io/gh/ajshajib/dolphin

Installation
------------

.. image:: https://img.shields.io/pypi/v/space-dolphin
   :alt: PyPI - Version

You can install ``dolphin`` using ``pip``. Run the following command:

.. code-block:: bash

    pip install space-dolphin

Alternatively, you can install the latest development version from GitHub as:

.. code-block:: bash

    git clone https://github.com/ajshajib/dolphin.git
    cd dolphin
    pip install .

See the `Quickstart guide <QUICKSTART.rst>`_ for instructions on setting up and running ``dolphin``.

Citation
--------

If you use ``dolphin`` in your research, please cite the ``dolphin`` paper Shajib et al. (2025). If you have used the ``"galaxy-quasar"`` fitting recipe, then additionally cite `Shajib et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.5649S/abstract>`_, and if you have used the ``"galaxy-galaxy"`` fitting recipe, then additionally cite `Shajib et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.2380S/abstract>`_.
