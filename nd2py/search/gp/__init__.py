# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from ...utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gp import GP
    from ...generator.eq.gplearn_generator import GPLearnGenerator

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "GP": (".gp", "search"),
    "GPLearnGenerator": ("...generator.eq.gplearn_generator", "search"),
})
