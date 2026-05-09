# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
The implementation of calculation functionality of Symbol, including:
- NumpyCalc: Calculate the value of a Symbol tree using numpy.
- TorchCalc: Calculate the value of a Symbol tree using torch (optional, requires nd2py[nn]).
"""

from .numpy_calc import NumpyCalc
from ...utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .torch_calc import TorchCalc

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "TorchCalc": (".torch_calc", "nn"),
})
