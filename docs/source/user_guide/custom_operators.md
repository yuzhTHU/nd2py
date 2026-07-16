(custom-operators-outside-nd2py)=
## Custom operators outside nd2py

You can define an operator and integrate it with nd2py visitors from an
application, notebook, or separate package. No modification of the nd2py
repository is required.

This page builds a unary `Cube` operator, adds NumPy evaluation and string
rendering, and makes it available to the parser.

### Define the Symbol

```python
import nd2py as nd


class Cube(nd.Symbol):
    """Compute the elementwise cube of one operand."""

    n_operands = 1
```

The inherited `Symbol.map_nettype` rule is sufficient because `Cube` preserves
the network scope of its operand. Override `map_nettype` when an operator maps
between scalar, node, and edge scope.

The generic tree, length, and nettype visitors already understand this class
through `n_operands`. Backend-specific semantics require registration.

### Add NumPy evaluation

```python
import numpy as np
from nd2py.core.calc.numpy_calc import NumpyCalc


def visit_Cube_numpy(self, node, *args, **kwargs):
    x = yield (node.operands[0], args, kwargs)
    return np.power(x, 3)


NumpyCalc.visit_Cube = visit_Cube_numpy
```

Visitor methods are generators. The yielded tuple asks the visitor to evaluate
the child with the current arguments. The value sent back into `x` is the
child's NumPy result.

### Add string rendering

```python
from nd2py.core.converter.string_printer import StringPrinter


def visit_Cube_string(self, node, *args, **kwargs):
    x = yield (node.operands[0], args, kwargs)
    if kwargs.get("raw"):
        return f"Cube({x})"
    if kwargs.get("latex"):
        return rf"\left({x}\right)^3"
    return f"cube({x})"


StringPrinter.visit_Cube = visit_Cube_string
```

Because dispatch uses the exact class name, the method must be named
`visit_Cube`.

### Protect existing registrations

Direct monkey patching is process-global. A small registration helper prevents
accidental replacement:

```python
def register_visitor(visitor_class, symbol_class, visit_method):
    method_name = f"visit_{symbol_class.__name__}"
    if hasattr(visitor_class, method_name):
        raise ValueError(f"{visitor_class.__name__}.{method_name} already exists")
    setattr(visitor_class, method_name, visit_method)


register_visitor(NumpyCalc, Cube, visit_Cube_numpy)
register_visitor(StringPrinter, Cube, visit_Cube_string)
```

For a reusable extension package, perform registration in an explicit setup
function rather than silently at import time.

### Build, evaluate, and parse

```python
x = nd.Variable("x")
expression = Cube(x) + 1

values = expression.eval({"x": np.array([1.0, 2.0, 3.0])})
# array([2., 9., 28.])

parsed = nd.parse(
    "cube(x) + 1",
    callables={"cube": Cube},
)
```

Passing `callables` makes the custom operator available without adding it to
`nd2py.core.symbols`.

### Which visitors need integration?

| Capability | Usually needs a custom method? | Reason |
|---|---:|---|
| Tree traversal | No | Uses `operands` generically |
| Tree printing | No | Generic structural output is sufficient |
| Copy | No for stateless operators | Reconstructs from operands and nettype |
| Nettype inference | No for scope-preserving arithmetic | Inherited mapping works |
| NumPy evaluation | Yes | Numerical semantics are operator-specific |
| EIC evaluation | Usually | Register an analytic derivative or NumPy semantics compatible with finite differences |
| Torch evaluation | Yes | Backend semantics are operator-specific |
| String rendering | Recommended | Generic output works but is less readable |
| Simplification | Only for special identities | Algebraic rules are operator-specific |
| Constant folding | Usually no | Generic evaluation works after a calculator is registered |

Stateful Symbols need more care than the stateless `Cube`: copying and any
transform that reconstructs the node must preserve their additional state.

### Complete runnable example

The following file is executed by the documentation test workflow:

```{literalinclude} ../../examples/custom_operator.py
:language: python
:linenos:
```
