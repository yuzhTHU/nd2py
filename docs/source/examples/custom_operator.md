This example adds an elementwise `cube(x)` operator without changing any file
inside nd2py. A custom Symbol defines the expression-tree structure, while
Visitor methods provide its NumPy and printing behavior.

```python
import numpy as np
import nd2py as nd

from nd2py.core.calc.numpy_calc import NumpyCalc
from nd2py.core.converter.string_printer import StringPrinter


class Cube(nd.Symbol):
    """Compute the elementwise cube of one operand."""

    n_operands = 1


def visit_Cube_numpy(self, node, *args, **kwargs):
    x = yield (node.operands[0], args, kwargs)
    return np.power(x, 3)


def visit_Cube_string(self, node, *args, **kwargs):
    x = yield (node.operands[0], args, kwargs)
    if kwargs.get("raw"):
        return f"Cube({x})"
    if kwargs.get("latex"):
        return rf"\left({x}\right)^3"
    return f"cube({x})"


NumpyCalc.visit_Cube = visit_Cube_numpy
StringPrinter.visit_Cube = visit_Cube_string
```

The Visitor protocol evaluates the operand by yielding it, then applies the
custom operation to the value sent back. Registration uses the exact class
name: `Cube` is dispatched to `visit_Cube`.

The new operator now composes with built-in Symbols:

```python
x = nd.Variable("x")
expression = Cube(x) + 1

print(expression)
# cube(x) + 1

result = expression.eval({
    "x": np.array([1.0, 2.0, 3.0]),
})
print(result)
# [ 2.  9. 28.]
```

It can also be made available to the parser without modifying nd2py's global
callable table:

```python
parsed = nd.parse(
    "cube(x) + 1",
    callables={"cube": Cube},
)
```

Direct monkey patching is process-global. Reusable extension packages should
perform registration in an explicit setup function and check that an existing
`visit_Cube` method is not being overwritten. The Guide contains a defensive
`register_visitor` helper and discusses which Visitors usually need custom
integration.
