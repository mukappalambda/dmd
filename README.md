# Python Implementation of Dynamic Mode Decomposition

This python code illustrates how to apply Dynamic Mode Decomposition (DMD) to univariate time series forecasting tasks.

Two examples are provided here.

For the explanation of DMD, please refer to this HackMD notes: [Dynamic Mode Decomposition](https://hackmd.io/@mklan/HyLXh7UH_).

The major reference is this arXiv article: [On Dynamic Mode Decomposition: Theory and Applications](https://arxiv.org/abs/1312.0041).

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
