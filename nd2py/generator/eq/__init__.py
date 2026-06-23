# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from ...utils.lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .enumerator import Enumerator
    from .metaai_generator import MetaAIGenerator
    from .gplearn_generator import GPLearnGenerator
    from .snip_generator import SNIPGenerator, SNIPGenerator2, SNIPGenerator3

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "Enumerator": (".enumerator", "all"),
    "MetaAIGenerator": (".metaai_generator", "all"),
    "GPLearnGenerator": (".gplearn_generator", "all"),
    "SNIPGenerator": (".snip_generator", "all"),
    "SNIPGenerator2": (".snip_generator", "all"),
    "SNIPGenerator3": (".snip_generator", "all"),
})
