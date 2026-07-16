# Examples

The examples on this page are complete tasks that can be copied into a script
or notebook. Use the page table of contents to move between examples.

## 1. Define variables

Create one variable directly:

```python
import nd2py as nd

x = nd.Variable("x")
```

Create several scalar variables at once:

```python
x, y, z = nd.variables("x y z")
```

Variables read values from a dictionary by name:

```python
data = {
    "x": [1.0, 2.0, 3.0],
    "y": [4.0, 5.0, 6.0],
}

print(x.eval(data))
# [1. 2. 3.]
```

Use `nettype` when a variable represents one value per node or edge:

```python
node_state = nd.Variable("node_state", nettype="node")
edge_weight = nd.Variable("edge_weight", nettype="edge")
```

## 2. Combine variables into an expression

Python arithmetic and nd2py functions build a Symbol tree:

```python
x, y = nd.variables("x y")

expression = 2 * x + nd.sin(y) - x**2 / 3
print(expression)
# 2 * x + sin(y) - x ** 2 / 3
```

Common functions include:

```python
expression = (
    nd.sin(x)
    + nd.cos(y)
    + nd.exp(-x)
    + nd.logabs(y)
    + nd.sqrtabs(x - y)
)
```

Python numbers embedded in an expression become fitable `Number` nodes. Use
`Constant` when a value must remain fixed:

```python
fitable_scale = nd.Number(2.0)
fixed_offset = nd.Constant(1.0)
expression = fitable_scale * x + fixed_offset
```

## 3. Evaluate an expression with NumPy

```python
import numpy as np

x, y = nd.variables("x y")
expression = x**2 + 2 * y

data = {
    "x": np.array([1.0, 2.0, 3.0]),
    "y": np.array([4.0, 5.0, 6.0]),
}

result = expression.eval(data)
print(result)
# [ 9. 14. 21.]
```

NumPy broadcasting applies normally:

```python
expression = x + y
result = expression.eval({
    "x": np.array([[1.0], [2.0]]),
    "y": np.array([10.0, 20.0, 30.0]),
})

assert result.shape == (2, 3)
```

## 4. Print an expression

The default form is intended for people:

```python
x = nd.Variable("x")
expression = nd.sin(2 * x + 1)

print(expression)
print(expression.to_str())
# sin(2 * x + 1)
```

### Tree form

```python
print(expression.to_tree())
# Sin (scalar)
# └ Add (scalar)
#   ├ Mul (scalar)
#   ┆ ├ 2 (scalar)
#   ┆ └ x (scalar)
#   └ 1 (scalar)
```

### LaTeX form

```python
latex = expression.to_str(latex=True)
print(latex)
# sin(2 \times x + 1)
```

The returned string can be passed to a notebook renderer or inserted into a
LaTeX document.

### Raw form

The raw form records constructors, nettypes, values, and fitability:

```python
raw = expression.to_str(raw=True)
print(raw)
# Sin(Number(2, "scalar", True) * Variable("x", "scalar") + Number(1, "scalar", True))

restored = nd.parse(raw)
np.testing.assert_allclose(restored.eval({"x": [1.0]}), expression.eval({"x": [1.0]}))
```

### Number formatting

```python
expression = nd.Number(1.234567) * x
print(expression.to_str(number_format=".3f"))
# 1.235 * x
```

## 5. Parse a formula from text

```python
expression = nd.parse("sin(2 * x) + y ** 2")

result = expression.eval({
    "x": np.array([0.0, 1.0]),
    "y": np.array([2.0, 3.0]),
})

print(result)
# [4.         9.90929743]
```

Provide predeclared variables when their nettypes matter:

```python
x = nd.Variable("x", nettype="node")
expression = nd.parse("2 * x + 1", variables={"x": x})
assert expression.nettype == "node"
```

## 6. Inspect and traverse the expression tree

```python
x = nd.Variable("x")
expression = nd.sin(2 * x + 1)

for node in expression.iter_preorder():
    print(type(node).__name__)
# Sin
# Add
# Mul
# Number
# Variable
# Number
```

Postorder is useful for bottom-up computations:

```python
nodes = list(expression.iter_postorder())
assert nodes[-1] is expression
```

## 7. Copy and replace a subexpression

`copy()` creates an independent expression tree:

```python
copied = expression.copy()
assert copied is not expression
```

Replace a selected node from a root expression:

```python
x = nd.Variable("x")
y = nd.Variable("y")
expression = nd.sin(x) + 1

x_node = next(
    node for node in expression.iter_preorder()
    if isinstance(node, nd.Variable) and node.name == "x"
)

expression = expression.replace(x_node, y)
print(expression)
# sin(y) + 1
```

## 8. Match an expression pattern

Pattern variables match arbitrary subexpressions:

```python
x = nd.Variable("x")
a = nd.Variable("a")

target = nd.sin(2 * x + 1)
bindings = target.match(nd.sin(a))

print(bindings["a"])
# 2 * x + 1
```

Repeated pattern names must match the same subexpression:

```python
pattern = a + a
assert (nd.sin(x) + nd.sin(x)).match(pattern) is not None
assert (nd.sin(x) + nd.cos(x)).match(pattern) is None
```

## 9. Fit numerical parameters

Fit the constants in a fixed formula with BFGS:

```python
x = nd.Variable("x")
expression = 0.5 * x + 0.5

X = {"x": np.linspace(-2.0, 2.0, 200)}
y = 3.0 * X["x"] + 2.0

fit = nd.BFGSFit(expression, fold_constant=True)
fit.fit(X, y)

print(fit.expression)
print(fit.loss_)
# 3.0000000371975344 * x + 2.0000000376510387
# 3.2997463432895325e-15

prediction = fit.predict(X)
assert np.mean((prediction - y) ** 2) < 1e-10
```

Inspect fitted Number nodes:

```python
parameters = [
    node.value
    for node in fit.expression.iter_preorder()
    if isinstance(node, nd.Number) and node.fitable
]
```

## 10. Work with network types

```python
x = nd.Variable("x", nettype="node")
scale = nd.Number(2.0, nettype="scalar")

expression = scale * x
assert expression.nettype == "node"
```

Map node values to edges using source and target indices:

```python
source_value = nd.sour(x)
target_value = nd.targ(x)

edge_list = ([0, 1, 2], [1, 2, 0])
data = {"x": np.array([10.0, 20.0, 30.0])}

np.testing.assert_array_equal(
    source_value.eval(data, edge_list=edge_list),
    np.array([10.0, 20.0, 30.0]),
)
np.testing.assert_array_equal(
    target_value.eval(data, edge_list=edge_list),
    np.array([20.0, 30.0, 10.0]),
)
```

## 11. Fit grouped parameters

```{include} examples/grouped_parameter.md
```

## 12. Define a custom operator

```{include} examples/custom_operator.md
```

## 13. Diagnose numerical exceptions

Ask NumPy evaluation to identify the exact subexpressions producing non-finite
values:

```python
x = nd.Variable("x")
expression = nd.log(x) + nd.inv(x)

result, exceptions = expression.eval(
    {"x": np.array([-1.0, 0.0, 1.0])},
    return_exceptions=True,
)

print(result)
print(*exceptions, sep="\n")
# [nan nan  1.]
# Subexpression log(x) produced non-finite values: 33.33% NaN, 0.00% Inf, 33.33% NegInf
# Subexpression 1 / x produced non-finite values: 0.00% NaN, 33.33% Inf, 0.00% NegInf
# Subexpression log(x) + 1 / x produced non-finite values: 66.67% NaN, 0.00% Inf, 0.00% NegInf

assert len(exceptions) == 3
assert "log(x)" in exceptions[0]
assert "1 / x" in exceptions[1]
```

Without `return_exceptions=True`, `eval` keeps its original return type and
returns only the numerical result.

## 14. Estimate numerical sensitivity with EIC

The Effective Information Criterion estimates the number of decimal digits
lost through numerical sensitivity:

```python
x = nd.Variable("x")

stable_eic = (x + 1).eval_eic({"x": np.array([1.0, 2.0, 3.0])})
unstable_eic, exceptions = (x - x).eval_eic(
    {"x": np.array([1.0, 2.0, 3.0])},
    return_exceptions=True,
)

print(stable_eic)
print(unstable_eic)
print(*exceptions, sep="\n")
# 0.09658807486020385
# inf
# The subexpression x - x exhibits elevated local numerical sensitivity,
# corresponding to an estimated loss of inf decimal digits of precision.

assert np.isfinite(stable_eic)
assert np.isinf(unstable_eic)
assert "x - x" in exceptions[0]
```
