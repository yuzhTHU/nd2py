from .core.context.check_nettype import no_nettype_check, set_nettype_check
from .core.context.set_fitable import no_set_fitable
from .core.context.warn_once import no_warn
from .core.symbols import *
from .core.printer.string_printer import StringPrinter
from .core.printer.tree_printer import TreePrinter
from .core.calc.numpy_calc import NumpyCalc
from .core.calc.torch_calc import TorchCalc
from .core.fit.fold_constant import FoldConstant
from .core.fit.bfgs_fit import BFGSFit
from .core.transform.split_by_add import SplitByAdd
from .core.transform.split_by_mul import SplitByMul
from .core.transform.fix_nettype import FixNetType
from . import utils
from .search.gp import GP, GPLearnGenerator
from .search.llmsr import LLMSR

Constant = lambda x, *args, **kwargs: Number(x, *args, **kwargs, fitable=False)
variables = lambda vars, *args, **kwargs: (
    [Variable(v, *args, **kwargs) for v in vars.split(" ") if v]
    if " " in vars
    else Variable(vars, *args, **kwargs)
)

add = lambda x, y: Add(x, y)
sub = lambda x, y: Sub(x, y)
mul = lambda x, y: Mul(x, y)
div = lambda x, y: Div(x, y)
pow = lambda x, y: Pow(x, y)
max = lambda x, y: Max(x, y)
min = lambda x, y: Min(x, y)
sin = lambda x: Sin(x)
cos = lambda x: Cos(x)
tan = lambda x: Tan(x)
sec = lambda x: Sec(x)
csc = lambda x: Csc(x)
cot = lambda x: Cot(x)
log = lambda x: Log(x)
logabs = lambda x: LogAbs(x)
exp = lambda x: Exp(x)
abs = lambda x: Abs(x)
neg = lambda x: Neg(x)
inv = lambda x: Inv(x)
sqrt = lambda x: Sqrt(x)
sqrtabs = lambda x: SqrtAbs(x)
pow2 = lambda x: Pow2(x)
pow3 = lambda x: Pow3(x)
arcsin = lambda x: Arcsin(x)
arccos = lambda x: Arccos(x)
arctan = lambda x: Arctan(x)
sinh = lambda x: Sinh(x)
cosh = lambda x: Cosh(x)
tanh = lambda x: Tanh(x)
sech = lambda x: Sech(x)
csch = lambda x: Csch(x)
sigmoid = lambda x: Sigmoid(x)
regular = lambda x, y: Regular(x, y)
sour = phi_s = lambda x: Sour(x)
targ = phi_t = lambda x: Targ(x)
aggr = rho = lambda x: Aggr(x)
rgga = lambda x: Rgga(x)
readout = lambda x: Readout(x)
