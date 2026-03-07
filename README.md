# nd2py

[![PyPI version](https://img.shields.io/pypi/v/nd2py.svg)](https://pypi.org/project/nd2py/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nd2py.svg)](https://pypi.org/project/nd2py/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://yuzhthu.github.io/nd2py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yuzhthu/nd2py/blob/main/LICENSE)

**nd2py** is a comprehensive and extensible Python framework designed for Symbolic Regression (SR) and the discovery of complex network dynamics. 

## 📖 About The Project

`nd2py` originally started as the official implementation for the paper *"Discovering network dynamics with neural symbolic regression"*. However, it has since evolved significantly beyond a single-paper repository. 

We have completely rewritten the underlying **Symbolic Engine** from the ground up to be highly efficient, modular, and extensible. Based on this robust core, `nd2py` now serves as a general-purpose symbolic regression library. The original algorithm from the paper (NDFormer) is now seamlessly integrated as one of the many powerful search modules within the framework.

Whether you are looking for classic genetic algorithms, modern Monte Carlo methods, or cutting-edge AI-driven approaches, `nd2py` provides a unified interface to explore the space of mathematical expressions.

## ✨ Core Features

* **Unified Symbolic Engine:** A custom-built, highly optimized core for symbolic expression representation, parsing, transformation, and fast evaluation (supporting both `numpy` and `torch` backends).
* **Diverse Search Algorithms:** Out-of-the-box support for multiple symbolic regression strategies:
    * 🧬 **`gp`**: Genetic Programming.
    * 🌳 **`mcts`**: Monte Carlo Tree Search.
    * 🧠 **`ndformer`**: Neural Symbolic Regression (from our original paper).
    * 🤖 **`llmsr`**: Large Language Model-guided Symbolic Regression.
* **Highly Extensible:** Designed with a clean architecture that makes it exceptionally easy to implement and benchmark your own custom symbolic regression algorithms.