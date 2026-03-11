# Python Implementation of Dynamic Mode Decomposition

[![Dependabot Updates](https://github.com/mukappalambda/dmd/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/mukappalambda/dmd/actions/workflows/dependabot/dependabot-updates)
[![Ruff](https://github.com/mukappalambda/dmd/actions/workflows/ruff.yml/badge.svg)](https://github.com/mukappalambda/dmd/actions/workflows/ruff.yml)

This Python package provides an implementation of Dynamic Mode Decomposition (DMD) specifically designed for univariate time series forecasting. DMD offers a data-driven approach for analyzing the dynamics of complex systems, making it suitable for forecasting and system identification applications.

## Table of Contents

- [Python Implementation of Dynamic Mode Decomposition](#python-implementation-of-dynamic-mode-decomposition)
  - [Install dmd](#install-dmd)
  - [Run Examples](#run-examples)
  - [Uninstall dmd](#uninstall-dmd)
  - [History](#history)

Two example scripts are included to demonstrate typical usage scenarios.

For an in-depth explanation of DMD, please refer to the following HackMD note: [Dynamic Mode Decomposition](https://hackmd.io/@mklan/HyLXh7UH_).

The primary reference for this implementation is the arXiv article: [On Dynamic Mode Decomposition: Theory and Applications](https://arxiv.org/abs/1312.0041).

---

## Install `dmd`

**Install from GitHub**

```bash
pip install git+https://github.com/mukappalambda/dmd.git
```

**Install from Source**

```bash
git clone https://github.com/mukappalambda/dmd.git
poetry build
find dist -name "*-$(poetry version -s)-*.whl" | xargs -I{} pip install {}
```

---

## Run Examples

```bash
cd examples
python dmd_example01.py
python dmd_example02.py
```

---

## Uninstall `dmd`

```bash
pip uninstall dmd -y
```

## History

- Tweak the HackMD note more readable and add the Python code inside that note.
