## Quick start

### Build and evaluate an expression

Symbols compose through ordinary Python arithmetic:

```python
import numpy as np
import nd2py as nd

x, y = nd.variables("x y")
expression = nd.sin(2 * x) + y**2

data = {
    "x": np.array([0.0, 0.5, 1.0]),
    "y": np.array([1.0, 2.0, 3.0]),
}

values = expression.eval(data)
print(expression)
print(values)
# sin(2 * x) + y ** 2
# [1.         4.84147098 9.90929743]
```

An expression is a tree. Inspect it with:

```python
print(expression.to_tree())
# Add (scalar)
# ├ Sin (scalar)
# ┆ └ Mul (scalar)
# ┆   ├ 2 (scalar)
# ┆   └ x (scalar)
# └ Pow2 (scalar)
#   └ y (scalar)
```

### Fit numerical constants

Numbers created through arithmetic are fitable by default. `BFGSFit` optimizes
them against observations while leaving the symbolic structure unchanged.

```python
x = nd.Variable("x")
model_expression = 0.5 * x + 0.5

X = {"x": np.linspace(-2.0, 2.0, 100)}
target = 3.0 * X["x"] + 2.0

fit = nd.BFGSFit(model_expression, fold_constant=True)
fit.fit(X, target)

print(fit.expression)
print(fit.loss_)
# 3.0000000371975344 * x + 2.0000000376510387
# 3.2997463432895325e-15
```

The last digits can vary slightly with SciPy versions; the fitted coefficient
and intercept should be close to 3 and 2, and the loss should be near zero.

Use {class}`nd2py.core.symbols.number.Number` for a fitable number and
{func}`nd2py.core.symbols.functions.Constant` for a fixed numerical constant.

### Parse an expression

```python
expression = nd.parse("sin(2 * x) + y ** 2")
```

Parsing uses the same Symbol classes as programmatic construction, so parsed
expressions support evaluation, printing, tree operations, and transforms.

### Where to go next

- Read {ref}`expressions-and-symbols` for the object model.
- Read {ref}`network-types` for graph-aware expressions.
- Read {ref}`parameter-fitting` for BFGS.
- Read {ref}`custom-operators-outside-nd2py` to extend nd2py outside the repository.
