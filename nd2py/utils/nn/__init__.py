# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from ..lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gnn import GNN
    from .positional_encoding import PositionalEncoding

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "GNN": (".gnn", "nn"),
    "PositionalEncoding": (".positional_encoding", "nn"),
})
