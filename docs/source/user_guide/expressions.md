(expressions-and-symbols)=
## Expressions and symbols

Every nd2py expression is a tree of {class}`~nd2py.core.symbols.symbol.Symbol`
objects. Leaves provide data or parameters, while internal nodes describe
operations.

### Core symbol kinds

| Symbol | Meaning | Depends on input data? | Fitable? |
|---|---|---:|---:|
| `Variable` | Named value read from the data mapping | Yes | No |
| `Number` | Numerical constant or parameter | No | Optional |
| `GroupedParameter` | Parameter selected by a categorical variable | Yes | Yes |
| Operators | Functions such as addition, sine, and aggregation | Through operands | No internal state |

```python
import nd2py as nd

x = nd.Variable("x", nettype="scalar")
a = nd.Number(2.0, fitable=True)
c = nd.Constant(1.0)
expression = a * x + c
```

Python numbers used inside an expression are automatically wrapped as
`Number` nodes:

```python
expression = 2.0 * x + 1.0
```

### Operands and ownership

Each Symbol declares `n_operands`. A unary operator has one operand, a binary
operator has two, and leaves have none. `Symbol` validates arity and links each
child to its parent during construction.

nd2py maintains a tree rather than a general directed acyclic graph. A
non-variable subexpression already owned by a parent is copied when inserted
elsewhere. This keeps replacement, type inference, and traversal semantics
unambiguous.

### String forms

```python
print(expression.to_str())
# 2 * x + Constant(1)

print(expression.to_str(raw=True))
# Number(2.0, "scalar", True) * Variable("x", "scalar") + Number(1.0, "scalar", False)

print(expression.to_str(latex=True))
# 2 \times x + Constant(1)

print(expression.to_tree())
# Add (scalar)
# ├ Mul (scalar)
# ┆ ├ 2.0 (scalar)
# ┆ └ x (scalar)
# └ Constant(1.0) (scalar)
```

The raw representation is intended for round trips through {func}`nd2py.parse`.

### Public operations

The methods exposed by a Symbol are a facade over specialized visitors. For
example, `eval()` invokes `NumpyCalc`, `to_str()` invokes `StringPrinter`, and
`copy()` invokes `GetCopy`. This separation lets the same expression tree have
multiple interpretations without putting every implementation inside Symbol.
