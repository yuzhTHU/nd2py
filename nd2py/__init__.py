from .core.context.check_nettype import no_nettype_check, set_nettype_check
from .core.context.set_fitable import no_set_fitable
from .core.context.warn_once import no_warn
from .core.symbols import *
from .core.functions import *
from .core.printer.string_printer import StringPrinter
from .core.printer.tree_printer import TreePrinter
from .core.calc.numpy_calc import NumpyCalc
from .core.calc.torch_calc import TorchCalc
from .core.fit.fold_constant import FoldConstant
from .core.fit.bfgs_fit import BFGSFit
from .core.transform.split_by_add import SplitByAdd
from .core.transform.split_by_mul import SplitByMul
from .core.transform.fix_nettype import FixNetType
from .core.parse.parser import parse
from . import utils
from .search.gp import GP, GPLearnGenerator
from .search.llmsr import LLMSR

Constant = lambda x, *args, **kwargs: Number(x, *args, **kwargs, fitable=False)
variables = lambda vars, *args, **kwargs: (
    [Variable(v, *args, **kwargs) for v in vars.split(" ") if v]
    if " " in vars
    else Variable(vars, *args, **kwargs)
)
