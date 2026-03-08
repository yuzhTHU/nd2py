# nd2py

<div align="center">

# nd2py

**Discovering network dynamics with neural symbolic regression.**

<p>
  <a href="https://www.nature.com/articles/s43588-025-00893-8">
    <img src="https://img.shields.io/badge/Published%20in-Nature%20Computational%20Science-004d3d" alt="Nature">
  </a>
  <!-- <a href="https://doi.org/10.1038/s43588-025-00893-8">
    <img src="https://img.shields.io/badge/DOI-10.1038/s43588--025--00893--8-blue" alt="DOI">
  </a> -->
  <a href="https://github.com/yuzhTHU/ND2">
    <img src="https://img.shields.io/badge/Refactored%20from-ND2-orange?logo=github" alt="Based on ND2">
  </a>
</p>

<p>
  <a href="https://pypi.org/project/nd2py/">
    <img src="https://img.shields.io/pypi/v/nd2py.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/nd2py/">
    <img src="https://img.shields.io/pypi/pyversions/nd2py.svg" alt="Python Versions">
  </a>
  <a href="https://yuzhthu.github.io/nd2py/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation">
  </a>
  <a href="https://github.com/yuzhthu/nd2py/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
</p>

<p>
  <a href="https://github.com/yuzhthu/nd2py">
    <img src="https://img.shields.io/badge/GitHub-nd2py-181717?logo=github" alt="GitHub Repo">
  </a>
  <a href="https://github.com/yuzhthu/nd2py">
    <img src="https://img.shields.io/github/last-commit/yuzhthu/nd2py" alt="Last Commit">
  </a>
  <a href="https://codecov.io/gh/yuzhthu/nd2py">
    <img src="https://codecov.io/gh/yuzhthu/nd2py/branch/main/graph/badge.svg" alt="codecov">
  </a>
</p>

</div>

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
