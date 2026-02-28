from functools import reduce
from . import symbols as symbols

__all__ = [
    "add", "sub", "mul", "div", "pow", "max", "min",
    "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan",
    "log", "logabs", "exp", "abs", "neg", "inv", "sqrt", "sqrtabs", "pow2", "pow3",
    "sinh", "cosh", "tanh", "sech", "csch", "sigmoid", "reg", "regular",
    "sour", "phi_s", "targ", "phi_t", "aggr", "rho", "rgga", "readout",
    "sum", "prod", "maximum", "minimum",
]

add = symbols.Add
sub = symbols.Sub
mul = symbols.Mul
div = symbols.Div
pow = symbols.Pow
max = symbols.Max
min = symbols.Min
sin = symbols.Sin
cos = symbols.Cos
tan = symbols.Tan
sec = symbols.Sec
csc = symbols.Csc
cot = symbols.Cot
log = symbols.Log
logabs = symbols.LogAbs
exp = symbols.Exp
abs = symbols.Abs
neg = symbols.Neg
inv = symbols.Inv
sqrt = symbols.Sqrt
sqrtabs = symbols.SqrtAbs
pow2 = symbols.Pow2
pow3 = symbols.Pow3
arcsin = symbols.Arcsin
arccos = symbols.Arccos
arctan = symbols.Arctan
sinh = symbols.Sinh
cosh = symbols.Cosh
tanh = symbols.Tanh
sech = symbols.Sech
csch = symbols.Csch
sigmoid = symbols.Sigmoid
reg = regular = symbols.Regular
sour = phi_s = symbols.Sour
targ = phi_t = symbols.Targ
aggr = rho = symbols.Aggr
rgga = symbols.Rgga
readout = symbols.Readout

def sum(*operands):
    return reduce(add, operands)

def prod(*operands):
    return reduce(mul, operands)

def maximum(*operands):
    return reduce(max, operands)

def minimum(*operands):
    return reduce(min, operands)
