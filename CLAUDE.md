# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```shell
conda create --prefix ./venv python=3.12
conda activate ./venv
pip install -e ".[all]"  # Installs nd2py with torch and torch_geometric
```

## Development Commands

```shell
# Run all tests
pytest

# Run only fast unit tests (default)
pytest

# Run slow integration tests
pytest --run-slow

# Run all tests including slow ones
pytest --run-all

# Run a specific test file
pytest tests/path/to/test.py

# Run tests matching a pattern
pytest -k pattern

# Install package in development mode
pip install -e .
```

## Architecture Overview

**nd2py** (Neural Discovery of Network Dynamics) is a symbolic regression package for discovering network dynamics.

### Package Structure

```
nd2py/
├── core/           # Core expression tree infrastructure
│   ├── symbols/    # Symbol types (Variable, Number, Add, Mul, functions)
│   ├── tree/       # Tree traversal utilities
│   ├── converter/  # Tree-to-string/from-list converters
│   ├── transform/  # Expression simplification, splitting, BFGS fitting
│   ├── calc/       # Numerical evaluation (numpy/torch backends)
│   └── nettype/    # Network type handling (node/edge/net)
├── search/         # Symbolic regression search algorithms
│   ├── mcts/       # Monte Carlo Tree Search
│   ├── gp/         # Genetic Programming
│   ├── llmsr/      # LLM-based Symbolic Regression
│   └── ndformer/   # Transformer-based approach
├── generator/      # Synthetic equation/data generators
├── dataset/        # Tokenization for ML models
└── utils/          # Utilities (plotting, logging, GPU selection)
```

### Key Concepts

- **Expression Trees**: Mathematical expressions are represented as trees with operators as internal nodes and variables/constants as leaves
- **Nettype System**: Distinguishes between node-level, edge-level, and network-level computations
- **Search Algorithms**: Multiple SR approaches available - MCTS, GP, LLM-SR, and NDFormer
- **Dual Backend**: Supports both NumPy and PyTorch for expression evaluation

### Usage Pattern

```python
import nd2py as nd
import numpy as np

# Define variables
x = nd.Variable('x', nettype='node')
y = nd.Variable('y', nettype='edge')

# Build expression tree
expr = x + nd.aggr(y * nd.sour(x))

# Run search
est = nd.MCTS(variables=[x, y], n_iter=3000, ...)
est.fit(X, y)
result = est.predict(X)
```

## TODO

### NDFormerMCTS - NDFormer-guided MCTS

**Status**: Implementation complete, pending testing

**Design**:
- `NDFormerMCTS` extends `MCTS` class, using PUCT instead of UCT
- NDFormer model provides prior probability P(s,a) for action selection
- Encoder memory is cached in `fit()` for efficient decoder calls

**Files**:
- `nd2py/search/ndformer/ndformer_search.py` - Main implementation
- `ndformer_search.debug.py` - Debug script for testing

**Usage**:
```python
from nd2py.search.ndformer import NDFormerMCTS, NDFormerConfig, NDFormerModel, NDFormerTokenizer

# Option 1: Pass NDFormer directly
model = NDformerModel(config)
search = NDFormerMCTS(variables=[x, y], ndformer=model, ndformer_tokenizer=tokenizer)

# Option 2: Load after initialization
search = NDFormerMCTS(variables=[x, y])
search.load_ndformer(checkpoint="path/to/model.pt", config=config)

search.fit(X, y)
```

**Pending**:
- Test with randomly initialized model on simple equation (e.g., `x + sin(y)`)
- Debug any issues with encoder/decoder interface
- Verify PUCT formula and policy prior calculation
