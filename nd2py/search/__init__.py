# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from ..utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import gp, mcts, llmsr, ndformer

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "gp": (".gp", "search"),
    "mcts": (".mcts", "search"),
    "llmsr": (".llmsr", "search"),
    "ndformer": (".ndformer", "search,nn"),
})
