import numpy as np
import nd2py as nd

x, y, z = nd.variables("x y z", nettype="scalar")
vars = {
    "x": np.random.rand(100, 1),
    "y": np.random.rand(100, 1),
    "z": np.random.rand(100, 1),
}

f = nd.sin(x + y) * z

est = nd.GP(
    variables=[x, y, z],
    n_iter=1000,
    random_state=42,
    binary=[nd.Add, nd.Mul],
    unary=[nd.Sin],
    log_per_sec=30,
    log_detailed_speed=True
)

X = vars
y = f.eval(vars)
est.fit(X, y)
assert np.abs(est.predict(X) - y).max() < 1e-10, f"{est.eqtree} != {f}"
