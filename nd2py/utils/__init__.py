# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from .tag2ansi import tag2ansi
from .log_exception import log_exception
from .classproperty import classproperty
from .fix_parser import add_minus_flags, add_negation_flags
from .lazy_loader import setup_lazy_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import nn
    from . import plot
    from .auto_gpu import AutoGPU
    from .attr_dict import AttrDict
    from .logger import init_logger
    from .utils import seed_all, softmax
    from .render_python import render_python
    from .render_markdown import render_markdown
    from .timing import NamedTimer, ParallelTimer, Timer
    from .metrics import MAE_score, MAPE_score, R2_score, RMSE_score, sMAPE_score

__getattr__, __dir__, __all__ = setup_lazy_imports(__name__, {
    "nn": (".nn", "nn"),
    "plot": (".plot", "all"),
    "AutoGPU": (".auto_gpu", "nn"),
    "AttrDict": (".attr_dict", "all"),
    "init_logger": (".logger", "all"),
    "seed_all": (".utils", "all"),
    "softmax": (".utils", "all"),
    "render_python": (".render_python", "all"),
    "render_markdown": (".render_markdown", "all"),
    "NamedTimer": (".timing", "all"),
    "ParallelTimer": (".timing", "all"),
    "Timer": (".timing", "all"),
    "MAE_score": (".metrics", "all"),
    "MAPE_score": (".metrics", "all"),
    "R2_score": (".metrics", "all"),
    "RMSE_score": (".metrics", "all"),
    "sMAPE_score": (".metrics", "all"),
})
