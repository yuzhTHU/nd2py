# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from functools import reduce
from .operands import *
from .variable import Variable
from .number import Number
from .empty import Empty
from ..nettype import NetType

__all__ = [
    "add", "sub", "mul", "div", "pow", "max", "min",
    "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan",
    "log", "logabs", "exp", "abs", "neg", "inv", "sqrt", "sqrtabs", "pow2", "pow3",
    "sinh", "cosh", "tanh", "sech", "csch", "sigmoid", "reg", "regular",
    "sour", "phi_s", "targ", "phi_t", "aggr", "rho", "rgga", "readout",
    "sum", "prod", "maximum", "minimum", "Constant", "variables"
]

add = Add
sub = Sub
mul = Mul
div = Div
pow = Pow
max = Max
min = Min
sin = Sin
cos = Cos
tan = Tan
sec = Sec
csc = Csc
cot = Cot
log = Log
logabs = LogAbs
exp = Exp
abs = Abs
neg = Neg
inv = Inv
sqrt = Sqrt
sqrtabs = SqrtAbs
pow2 = Pow2
pow3 = Pow3
arcsin = Arcsin
arccos = Arccos
arctan = Arctan
sinh = Sinh
cosh = Cosh
tanh = Tanh
sech = Sech
csch = Csch
sigmoid = Sigmoid
reg = regular = Regular
sour = phi_s = Sour
targ = phi_t = Targ
aggr = rho = Aggr
rgga = Rgga
readout = Readout

def sum(*operands):
    """Combine operands into an additive expression.

    Examples:
        >>> import nd2py as nd
        >>> x, y = nd.variables("x y")
        >>> nd.sum(x, y, 1).to_str()
        'x + y + 1'
    """
    return reduce(add, operands)

def prod(*operands):
    """Combine operands into a multiplicative expression.

    Examples:
        >>> import nd2py as nd
        >>> x, y = nd.variables("x y")
        >>> nd.prod(2, x, y).to_str()
        '2 * x * y'
    """
    return reduce(mul, operands)

def maximum(*operands):
    """Return the elementwise maximum of all operands.

    Examples:
        >>> import nd2py as nd
        >>> x, y = nd.variables("x y")
        >>> float(nd.maximum(x, y).eval({"x": 1.0, "y": 2.0}))
        2.0
    """
    return reduce(max, operands)

def minimum(*operands):
    """Return the elementwise minimum of all operands.

    Examples:
        >>> import nd2py as nd
        >>> x, y = nd.variables("x y")
        >>> float(nd.minimum(x, y).eval({"x": 1.0, "y": 2.0}))
        1.0
    """
    return reduce(min, operands)


def Constant(value, nettype: NetType = "scalar") -> Number:
    """Create a fixed numerical constant.

    Args:
        value: Numerical value of the constant.
        nettype: Network type of the constant. Defaults to ``"scalar"``.

    Examples:
        >>> import nd2py as nd
        >>> constant = nd.Constant(2.0)
        >>> constant.fitable
        False
    """
    return Number(value, nettype=nettype, fitable=False)

def variables(vars, *args, **kwargs):
    """Create one variable or a list of space-separated variables.

    Args:
        vars: One variable name or several names separated by spaces.
        *args: Additional positional arguments passed to :class:`Variable`.
        **kwargs: Additional keyword arguments passed to :class:`Variable`.

    Examples:
        >>> import nd2py as nd
        >>> x, y = nd.variables("x y")
        >>> x.name, y.name
        ('x', 'y')
    """
    if isinstance(vars, str) and " " in vars:
        return [Variable(v, *args, **kwargs) for v in vars.split(" ") if v]
    else:
        return Variable(vars, *args, **kwargs)
