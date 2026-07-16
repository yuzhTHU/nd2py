(parameter-fitting)=
## Parameter fitting

{class}`~nd2py.core.transform.bfgs_fit.BFGSFit` optimizes fitable numerical
parameters for a fixed symbolic structure.

```python
import numpy as np
import nd2py as nd

x = nd.Variable("x")
expression = 0.5 * x + 0.5

X = {"x": np.linspace(-1.0, 1.0, 100)}
y = 3.0 * X["x"] + 2.0

fit = nd.BFGSFit(expression, fold_constant=True)
fit.fit(X, y)

print(fit.expression)
print(fit.loss_, fit.success_)
# 3.000000175954602 * x + 1.9999999661591596
# 1.1673694793736501e-14 True
```

Exact final digits can vary with SciPy versions. The meaningful checks are
that the parameters approach 3 and 2, the loss is near zero, and fitting
reports success.

### Fittable and fixed constants

`Number(..., fitable=True)` participates in optimization. The default depends
on the `set_fitable` context. `Constant(...)` explicitly creates a fixed
Number.

### Constant folding

With `fold_constant=True`, BFGS first combines fitable constant-only subtrees,
then evaluates data-only subtrees once. The resulting loss expression is a
temporary tree. After optimization, BFGS explicitly writes every fitted value
back to `fit.expression`; it does not rely on the temporary and retained trees
sharing node identity.

### Parameter arrays

Parameters are flattened before SciPy optimization and restored to their
original shapes afterward. This supports scalar Numbers, array-valued Numbers,
and GroupedParameters in one expression.

### Estimator attributes

After `fit`, inspect:

- `expression`: retained expression containing fitted values;
- `loss_`: final mean squared error;
- `success_`: SciPy optimizer success flag;
- `message_`: optimizer termination message;
- `n_iter_`: number of optimizer iterations.
