import nd2py as nd2
import numpy as np


prompt = """Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator system with driving force, given data on position, and velocity. 

You should generate `def equation(...) directly, without any additional comments or explanations."""


def evaluate(x: np.ndarray, v: np.ndarray, y: np.ndarray, maxn_params=10) -> float:
    """Evaluate the equation on data observations."""
    # Optimize parameters based on data
    from scipy.optimize import minimize

    def loss(params):
        y_pred = equation(x, v, params)
        return np.mean((y_pred - y) ** 2)

    loss_partial = lambda params: loss(params)
    result = minimize(loss_partial, [1.0] * maxn_params, method="BFGS")

    # Return evaluation score
    optimized_params = result.x
    loss = result.fun
    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Mathematical function for acceleration in a damped nonlinear oscillator
    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying the mathematical function to the inputs.
    """
    dv = params[0] * x + params[1] * v + params[2]
    return dv


namespace = {"np": np}
est = nd2.LLMSR(
    prompt=prompt,
    eval_program=evaluate,
    seed_program=equation,
    namespace=namespace,
    log_per_iter=1,
    save_path=None,
    log_detailed_speed=True,
)

N = 100
x = np.random.random(N)
v = np.random.random(N)
y = 1.0 * np.sin(2.0 * x) + 0.5 * v**2 + 0.1 * x
est.fit({"x": x, "v": v, "y": y})
# print(est.best_model)
