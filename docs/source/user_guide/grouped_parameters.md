## Grouped parameters

{class}`~nd2py.core.symbols.grouped_parameter.GroupedParameter` associates one
fitable scalar with each distinct label of a categorical variable.

Suppose the same linear structure applies to several groups but each group has
its own slope:

```python
import numpy as np
import nd2py as nd

data = {
    "s": np.array(["group1", "group1", "group2", "group2", "group3", "group3"]),
    "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    "y": np.array([1.0, 2.0, 6.0, 8.0, 15.0, 18.0]),
}

s = nd.Variable("s")
x = nd.Variable("x")
slope = nd.GroupedParameter(s, default=1.0)
expression = slope * x

fit = nd.BFGSFit(expression)
fit.fit(data, data["y"])

print(fit.expression.operands[0].value_dict)
# {'group1': 1.0000001749568208, 'group2': 1.9999999888181819,
#  'group3': 2.99999999086196}
```

The final digits can vary with SciPy versions; the meaningful fitted values
are approximately 1, 2, and 3.

### Category binding

Labels are bound in first-seen order. Previously unseen labels are appended to
`group_labels`, added to `label_to_index`, and initialized with `default`.
Evaluation therefore has defaultdict-like semantics:

```python
parameter = nd.GroupedParameter(s, value={"known": 2.0}, default=0.5)
parameter.eval({"s": np.array(["known", "new"])})

assert parameter.value_dict == {"known": 2.0, "new": 0.5}
```

New categories encountered after fitting use `default`; they have not been
optimized from training observations.

### Representation

One grouped parameter prints as `alpha[s]`. Multiple grouped parameters in one
expression are numbered by preorder occurrence:

```text
alpha^(1)[s] * x + alpha^(2)[s]
```

The LaTeX form is `\alpha^{(1)}_{s}`. Use
`grouped_parameter_symbol="theta"` with `to_str` to change the displayed symbol.

### Backend support

GroupedParameter currently supports NumPy evaluation and BFGS fitting. Torch
evaluation raises a deliberate `NotImplementedError` because string categories
must first be encoded into integer tensor indices.
