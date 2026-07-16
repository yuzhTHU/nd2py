This example fits `y = alpha[s] * x`, where every value of `s` selects a
different trainable slope.

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
expression = nd.GroupedParameter(s, default=1.0) * x

fit = nd.BFGSFit(expression)
fit.fit(data, data["y"])

fitted_parameter = fit.expression.operands[0]
print(fitted_parameter.value_dict)
# {'group1': 1.0000001749568208, 'group2': 1.9999999888181819,
#  'group3': 2.99999999086196}

prediction = fit.predict(data)
mse = np.mean((prediction - data["y"]) ** 2)
assert mse < 1e-10
```

Expected fitted values are approximately:

```python
{
    "group1": 1.0,
    "group2": 2.0,
    "group3": 3.0,
}
```
