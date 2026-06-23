# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from ...utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mcts import MCTS, Node

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "MCTS": (".mcts", "search"),
    "Node": (".mcts", "search"),
})
