from sympy import Function, Symbol
from functools import reduce

class Aggr(Function):
    """
    Aggregate Operator: turn an edge feature into a node feature
    """
    def __init__(self, *args):
        super(Aggr, self).__init__()
        if len(self.args) == 3:
            x, G, A = self._args
            assert str(G) == 'G' and str(A) == 'A'
            self._args = (x,)
        assert len(self.args) == 1, args

    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\rho({})'.format(_m)
        if exp is not None:
            base = r'\rho^{{{}}}({})'.format(exp, _m)
        return base

    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return '\033[0;33maggr\033[0m({})'.format(_m)
    
    def _eval_expand_multinomial(self, **hints):
        m = self.args[0]
        if m.is_Add:
            return reduce(lambda x, y: x+y, [self.func(i) for i in m.args], 0)
        if m.is_Mul:
            coefficients = [i for i in m.args if i.is_number]
            variables = [i for i in m.args if not i.is_number]
            coeffieient = reduce(lambda x, y: x*y, coefficients, 1)
            function = self.func(reduce(lambda x, y: x*y, variables, 1))
            return coeffieient * function
        else:
            return self
    
class Sour(Function):
    """
    Source Operator: turn a node feature into an edge feature
    """
    def __init__(self, *args):
        super(Sour, self).__init__()
        if len(self.args) == 3:
            x, G, A = self._args
            assert str(G) == 'G' and str(A) == 'A'
            self._args = (x,)
        assert len(self.args) == 1, args
    
    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\phi_{{s}}({})'.format(_m)
        if exp is not None:
            base = r'\phi^{{{}}}_{{s}}({})'.format(exp, _m)
        return base
    
    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return '\033[0;33msour\033[0m({})'.format(_m)  
    
    def _eval_expand_multinomial(self, **hints):
        m = self.args[0]
        if m.is_Add:
            return reduce(lambda x, y: x+y, [self.func(i) for i in m.args], 0)
        if m.is_Mul:
            coefficients = [i for i in m.args if i.is_number]
            variables = [i for i in m.args if not i.is_number]
            coeffieient = reduce(lambda x, y: x*y, coefficients, 1)
            function = self.func(reduce(lambda x, y: x*y, variables, 1))
            return coeffieient * function
        else:
            return self

class Targ(Function):
    """
    Termination Operator: turn a node feature into an edge feature
    """
    def __init__(self, *args):
        super(Targ, self).__init__()
        if len(self.args) == 3:
            x, G, A = self._args
            assert str(G) == 'G' and str(A) == 'A'
            self._args = (x,)
        assert len(self.args) == 1, args
    
    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\phi_{{t}}({})'.format(_m)
        if exp is not None:
            base = r'\phi^{{{}}}_{{t}}({})'.format(exp, _m)
        return base
    
    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return '\033[0;33mtarg\033[0m({})'.format(_m)  
    
    def _eval_expand_multinomial(self, **hints):
        m = self.args[0]
        if m.is_Add:
            return reduce(lambda x, y: x+y, [self.func(i) for i in m.args], 0)
        if m.is_Mul:
            coefficients = [i for i in m.args if i.is_number]
            variables = [i for i in m.args if not i.is_number]
            coeffieient = reduce(lambda x, y: x*y, coefficients, 1)
            function = self.func(reduce(lambda x, y: x*y, variables, 1))
            return coeffieient * function
        else:
            return self

class Rgga(Function):
    """
    Reverse Aggregate Operator: turn an edge feature into a node feature
    """
    def __init__(self, *args):
        super(Rgga, self).__init__()
        if len(self.args) == 3:
            x, G, A = self._args
            assert str(G) == 'G' and str(A) == 'A'
            self._args = (x,)
        assert len(self.args) == 1, args

    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\rho^{{T}}({})'.format(_m)
        if exp is not None:
            base = r'\rho^{{T{}}}({})'.format(exp, _m)
        return base

    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return '\033[0;33mrgga\033[0m({})'.format(_m)
    
    def _eval_expand_multinomial(self, **hints):
        m = self.args[0]
        if m.is_Add:
            return reduce(lambda x, y: x+y, [self.func(i) for i in m.args], 0)
        if m.is_Mul:
            coefficients = [i for i in m.args if i.is_number]
            variables = [i for i in m.args if not i.is_number]
            coeffieient = reduce(lambda x, y: x*y, coefficients, 1)
            function = self.func(reduce(lambda x, y: x*y, variables, 1))
            return coeffieient * function
        else:
            return self


class Sigmoid(Function):
    def __init__(self, *args):
        super(Sigmoid, self).__init__()
        assert len(self.args) == 1, args
    
    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\sigma({})'.format(_m)
        if exp is not None:
            base = r'\sigma^{{{}}}({})'.format(exp, _m)
        return base
    
    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return 'sigmoid({})'.format(_m)
    

class Regular(Function):
    def __init__(self, *args):
        super(Regular, self).__init__()
        assert len(self.args) == 2, args
    
    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        n = self.args[1]
        _n = printer._print(n)
        base = r'\mathrm{{Reg}}({},{})'.format(_m, _n)
        if exp is not None:
            base = r'\mathrm{{Reg}}^{{{}}}({},{})'.format(exp, _m, _n)
        return base
    
    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        n = self.args[1]
        _n = printer._print(n)
        return 'regular({},{})'.format(_m, _n)

class LogAbs(Function):
    def __init__(self, *args):
        super(LogAbs, self).__init__()
        assert len(self.args) == 1, args
    
    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\log\left|{}\right|'.format(_m)
        if exp is not None:
            base = r'\log^{{{}}}\left|{}\right|'.format(exp, _m)
        return base
    
    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return 'logabs({})'.format(_m)

class SqrtAbs(Function):
    def __init__(self, *args):
        super(SqrtAbs, self).__init__()
        assert len(self.args) == 1, args
    
    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\sqrt{{\left|{}\right|}}'.format(_m)
        if exp is not None:
            base = r'\left|{}\right|^{{\frac{{{}}}{{2}}}}'.format(_m, exp)
        return base
    
    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return 'sqrtabs({})'.format(_m)
    

class Abs(Function):
    """ 覆盖掉 sympy 中的 abs 函数，那个总是会考虑复数的问题"""
    def __init__(self, *args):
        super(Abs, self).__init__()
        assert len(self.args) == 1, args
    
    def _latex(self, printer, exp=None):
        m = self.args[0]
        _m = printer._print(m)
        base = r'\left|{}\right|'.format(_m)
        if exp is not None:
            base = r'\left|{}\right|^{{{}}}'.format(_m, exp)
        return base

    def _sympystr(self, printer):
        m = self.args[0]
        _m = printer._print(m)
        return 'abs({})'.format(_m)