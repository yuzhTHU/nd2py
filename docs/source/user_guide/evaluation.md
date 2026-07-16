## Evaluation and visitors

nd2py evaluates and transforms expressions with visitors. A visitor dispatches
on the exact runtime class name:

```text
Add              -> visit_Add
Variable         -> visit_Variable
GroupedParameter -> visit_GroupedParameter
```

If a matching method is unavailable, the visitor uses `generic_visit`.

### NumPy evaluation

```python
result = expression.eval(
    vars={"x": x_values},
    edge_list=(sources, targets),
    num_nodes=number_of_nodes,
    use_eps=1e-8,
)
```

{class}`~nd2py.core.calc.numpy_calc.NumpyCalc` reads variables from `vars` and
applies NumPy operations node by node. Sample dimensions are ordinary NumPy
dimensions; nettype describes graph scope rather than array dtype or a general
shape system.

Set `return_exceptions=True` to locate subexpressions that produce NaN or
positive or negative infinity. The normal return value is unchanged by
default; when diagnostics are enabled, evaluation returns
`(result, exceptions)`:

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
```

Each diagnostic reports the offending subexpression and the fraction of its
values that are NaN, Inf, or NegInf.

### Effective Information Criterion

`expression.eval_eic(data)` estimates the number of decimal digits effectively
lost through numerical sensitivity in the expression tree. Larger values
indicate less stable structure; zero means no estimated loss at the sampled
inputs.

```python
x = nd.Variable("x")
unstable = x - x

eic, exceptions = unstable.eval_eic(
    {"x": np.array([1.0, 2.0])},
    return_exceptions=True,
)

assert np.isinf(eic)
assert "x - x" in exceptions[0]

print(eic)
print(*exceptions, sep="\n")
# inf
# The subexpression x - x exhibits elevated local numerical sensitivity,
# corresponding to an estimated loss of inf decimal digits of precision.
```

Common elementwise operators use analytic derivatives. Other supported NumPy
operators fall back to central finite differences controlled by
`perturbation`. `exception_threshold` controls which local digit-loss values
are reported; it does not change the returned EIC.

### PyTorch evaluation

```python
result = expression.eval_torch(vars=data, device="cuda")
```

{class}`~nd2py.core.calc.torch_calc.TorchCalc` follows the same expression tree
with PyTorch operations. Not every specialized Symbol necessarily supports
both backends. In particular, `GroupedParameter` currently supports NumPy only.

### Visitor protocol

Visitor methods are generators. They yield a child together with the arguments
used to visit it, then return the current node's result:

```python
def visit_Square(self, node, *args, **kwargs):
    x = yield (node.operands[0], args, kwargs)
    return x**2
```

The explicit stack in the base Visitor avoids Python recursion for deep trees.
See {ref}`custom-operators-outside-nd2py` for a complete extension example.
