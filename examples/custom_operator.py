"""Define and register an nd2py operator without modifying nd2py itself."""

import numpy as np
import nd2py as nd

from nd2py.core.calc.numpy_calc import NumpyCalc
from nd2py.core.converter.string_printer import StringPrinter


class Cube(nd.Symbol):
    """Compute the elementwise cube of one operand."""

    n_operands = 1

    @classmethod
    def register(cls):

        def register_visitor(visitor_class, symbol_class, visit_method):
            """Register a visit method while protecting existing registrations."""
            method_name = f"visit_{symbol_class.__name__}"
            if not hasattr(visitor_class, method_name):
                setattr(visitor_class, method_name, visit_method)

        def visit_Cube_string(self, node, *args, **kwargs):
            """Render Cube in readable and LaTeX expressions."""
            x = yield (node.operands[0], args, kwargs)
            if kwargs.get("raw"):
                return f"Cube({x})"
            if kwargs.get("latex"):
                return rf"\left({x}\right)^3"
            return f"cube({x})"

        register_visitor(StringPrinter, cls, visit_Cube_string)

        def visit_Cube_numpy(self, node, *args, **kwargs):
            """Evaluate Cube with NumPy."""
            x = yield (node.operands[0], args, kwargs)
            return np.power(x, 3)

        register_visitor(NumpyCalc, cls, visit_Cube_numpy)


Cube.register()

x = nd.Variable("x")
expression = Cube(x) + 1

# StringPrinter
print(expression)

# NumpyCalc
data = {"x": np.array([1.0, 2.0, 3.0])}
print(expression.eval(data))

# Parse
parsed = nd.parse("cube(x) + 1", callables={"cube": Cube})
print(parsed.eval(data))
