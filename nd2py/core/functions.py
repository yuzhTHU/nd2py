from .symbols import *

# fmt: off
__all__ = [
    'add', 'sub', 'mul', 'div', 'pow', 'max', 'min',
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'log',
    'inv', 'sqrt', 'sqrtabs', 'pow2', 'pow3',
    'logabs', 'exp', 'sigmoid', 'regular',
    'abs', 'neg', 'arcsin', 'arccos', 'arctan', 
    'sinh', 'cosh', 'tanh', 'sech', 'csch',
    'phi_s', 'phi_t', 'rho',
    'sour', 'targ', 'aggr', 'rgga', 'readout',
]
# fmt: on

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
