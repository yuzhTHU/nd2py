# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
The implementation of calculation functionality of Symbol, including:
- NumpyCalc: Calculate the value of a Symbol tree using numpy.
- TorchCalc: Calculate the value of a Symbol tree using torch.
"""

from .numpy_calc import NumpyCalc
from .torch_calc import TorchCalc
