# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .attr_dict import *
from .logger import *
from .metrics import *
from .plot import *
from .timing import *
from .utils import *
from .tag2ansi import tag2ansi
from .log_exception import log_exception
from .classproperty import classproperty
from .render_python import render_python
from .render_markdown import render_markdown
from .fix_parser import add_minus_flags, add_negation_flags
from .lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auto_gpu import AutoGPU
    from . import nn

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "AutoGPU": (".auto_gpu", "nn"),
    "nn": (".nn", "nn"),
})
