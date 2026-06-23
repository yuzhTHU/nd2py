# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from ...utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gmm_generator import GMMGenerator
    from .subeq_generator import SubeqGenerator

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "GMMGenerator": (".gmm_generator", "all"),
    "SubeqGenerator": (".subeq_generator", "all"),
})
