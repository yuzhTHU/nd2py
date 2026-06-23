# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .fix_nettype import FixNetType
from .fold_constant import FoldConstant
from .simplify import Simplify
from .split_by_add import SplitByAdd
from .split_by_mul import SplitByMul
from ...utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bfgs_fit import BFGSFit

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "BFGSFit": (".bfgs_fit", "all"),
})
