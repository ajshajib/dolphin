.. :changelog:

History
-------

1.0.0 (2025-03-28)
++++++++++++++++++

* Added a fully automated modeling workflow powered by AI
* Added feature to include satellite(s) in the lens model
* Notebook for Neural Network Training and Testing by @pensive-aristocrat in https://github.com/ajshajib/dolphin/pull/93
* Simplified main deflector's light profile specifications in the config.yaml for multiband fitting
* Updated documentation
* Updated Readme and added a Quickstart Guide

0.0.2 (2024-09-05)
++++++++++++++++++
* Add SIE lens model support by @ajshajib in https://github.com/ajshajib/dolphin/pull/32
* Support elliptical mask shapes by @ajshajib in https://github.com/ajshajib/dolphin/pull/46
* Make compatible with latest lenstronomy=1.12.0 and Python=3.11 by @ajshajib in https://github.com/ajshajib/dolphin/pull/48

**Full Changelog**: https://github.com/ajshajib/dolphin/compare/v0.0.1...v0.0.2

0.0.1 (2024-03-23)
++++++++++++++++++
This is the version of `dolphin` used in [Tan et al. (2024)](https://ui.adsabs.harvard.edu/abs/2023arXiv231109307T/abstract).

Supported features:

- Multi-band lens modeling with `lenstronomy` v1.11.5 or below
- Semi-automated lens modeling for a sample based on user-provided configuration in yaml files
- Easy setup to perform optimization/MCMC using HPCC for a large sample
- Initial fitting recipe to achieve faster convergence for galaxy-galaxy lenses


