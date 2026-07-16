## Installation

nd2py requires Python 3.12 or newer.

### Install from PyPI

```bash
python -m pip install nd2py
```

Install optional neural-network or search dependencies when needed:

```bash
python -m pip install "nd2py[nn]"
python -m pip install "nd2py[search]"
python -m pip install "nd2py[all]"
```

### Install for development

```bash
git clone https://github.com/yuzhTHU/nd2py.git
cd nd2py
python -m pip install -e ".[dev]"
pytest
```

The core symbolic engine and NumPy evaluator do not require the optional
PyTorch stack.
