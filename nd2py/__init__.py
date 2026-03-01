from .core.context.nettype_inference import no_nettype_inference, set_nettype_inference
from .core.context.set_fitable import no_set_fitable
from .core.context.copy_value import no_copy_value
from .core.context.warn_once import no_warn
from .core.symbols import *
from .core.converter import parse, StringPrinter, TreePrinter
from .core.calc import NumpyCalc, TorchCalc
from .core.transform import FoldConstant, BFGSFit, SplitByAdd, SplitByMul, FixNetType
from . import utils
from .search.gp import GP, GPLearnGenerator
from .search.mcts import MCTS
from .search.llmsr import LLMSR

Constant = lambda x, *args, **kwargs: Number(x, *args, **kwargs, fitable=False)
variables = lambda vars, *args, **kwargs: (
    [Variable(v, *args, **kwargs) for v in vars.split(" ") if v]
    if " " in vars
    else Variable(vars, *args, **kwargs)
)
