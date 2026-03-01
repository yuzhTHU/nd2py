# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from functools import reduce
from .operands import *

__all__ = [
    "add", "sub", "mul", "div", "pow", "max", "min",
    "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan",
    "log", "logabs", "exp", "abs", "neg", "inv", "sqrt", "sqrtabs", "pow2", "pow3",
    "sinh", "cosh", "tanh", "sech", "csch", "sigmoid", "reg", "regular",
    "sour", "phi_s", "targ", "phi_t", "aggr", "rho", "rgga", "readout",
    "sum", "prod", "maximum", "minimum",
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
    return reduce(add, operands)

def prod(*operands):
    return reduce(mul, operands)

def maximum(*operands):
    return reduce(max, operands)

def minimum(*operands):
    return reduce(min, operands)
