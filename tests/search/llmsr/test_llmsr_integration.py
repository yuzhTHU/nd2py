"""Integration tests for LLMSR symbolic regression capability."""
import os
import pytest
import numpy as np
from dotenv import load_dotenv
from nd2py.search.llmsr import LLMSR

load_dotenv()

# Skip all tests if SILICONFLOW_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("SILICONFLOW_API_KEY"),
    reason="Requires SILICONFLOW_API_KEY environment variable"
)


@pytest.mark.slow
class TestLLMSRIntegration:
    """Integration tests for LLMSR on simple symbolic regression problems."""

    def test_linear_regression(self):
        """Test LLMSR on a simple linear equation: y = 2*x + 1"""
        np.random.seed(42)
        N = 50
        x = np.linspace(-1, 1, N)
        y = 2 * x + 1

        def evaluate(x: np.ndarray, y: np.ndarray, maxn_params=5) -> float:
            from scipy.optimize import minimize

            def loss(params):
                y_pred = equation(x, params)
                return np.mean((y_pred - y) ** 2)

            result = minimize(lambda p: loss(p), [1.0] * maxn_params, method="BFGS")
            loss_val = result.fun
            return -loss_val if np.isfinite(loss_val) else float('-inf')

        def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            """Seed: linear model."""
            return params[0] * x + params[1]

        est = LLMSR(
            prompt="Find the mathematical function skeleton that best fits the data. "
                   "Return only the def equation(...) function.",
            eval_program=evaluate,
            seed_program=equation,
            namespace={"np": np},
            n_islands=3,
            n_iter=5,
            programs_per_prompt=1,
            save_path=None,
        )

        data = {"x": x, "y": y}
        est.fit(data)

        # Just verify it runs without error and produces output
        assert est.best_model is not None
        assert "def equation" in est.best_model.program

    def test_polynomial_regression(self):
        """Test LLMSR on a quadratic equation: y = x^2 + 2*x + 1"""
        np.random.seed(42)
        N = 50
        x = np.linspace(-2, 2, N)
        y = x**2 + 2 * x + 1

        def evaluate(x: np.ndarray, y: np.ndarray, maxn_params=5) -> float:
            from scipy.optimize import minimize

            def loss(params):
                y_pred = equation(x, params)
                return np.mean((y_pred - y) ** 2)

            result = minimize(lambda p: loss(p), [1.0] * maxn_params, method="BFGS")
            loss_val = result.fun
            return -loss_val if np.isfinite(loss_val) else float('-inf')

        def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            """Seed: linear model (LLMSR should discover quadratic)."""
            return params[0] * x + params[1]

        est = LLMSR(
            prompt="Find the mathematical function skeleton that best fits the data. "
                   "Return only the def equation(...) function.",
            eval_program=evaluate,
            seed_program=equation,
            namespace={"np": np},
            n_islands=3,
            n_iter=5,
            programs_per_prompt=1,
            save_path=None,
        )

        data = {"x": x, "y": y}
        est.fit(data)

        assert est.best_model is not None
        assert "def equation" in est.best_model.program

    def test_trigonometric_regression(self):
        """Test LLMSR on a trigonometric equation: y = x + sin(x)"""
        np.random.seed(42)
        N = 50
        x = np.linspace(-np.pi, np.pi, N)
        y = x + np.sin(x)

        def evaluate(x: np.ndarray, y: np.ndarray, maxn_params=5) -> float:
            from scipy.optimize import minimize

            def loss(params):
                y_pred = equation(x, params)
                return np.mean((y_pred - y) ** 2)

            result = minimize(lambda p: loss(p), [1.0] * maxn_params, method="BFGS")
            loss_val = result.fun
            return -loss_val if np.isfinite(loss_val) else float('-inf')

        def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            """Seed: simple linear model."""
            return params[0] * x

        est = LLMSR(
            prompt="Find the mathematical function skeleton that best fits the data. "
                   "Return only the def equation(...) function. "
                   "Consider using trigonometric functions like sin, cos if they improve the fit.",
            eval_program=evaluate,
            seed_program=equation,
            namespace={"np": np, "sin": np.sin, "cos": np.cos},
            n_islands=3,
            n_iter=5,
            programs_per_prompt=1,
            save_path=None,
        )

        data = {"x": x, "y": y}
        est.fit(data)

        assert est.best_model is not None
        assert "def equation" in est.best_model.program

    def test_multivariable_regression(self):
        """Test LLMSR on a multivariable equation: z = x + 2*y"""
        np.random.seed(42)
        N = 50
        x = np.random.random(N)
        y = np.random.random(N)
        z = x + 2 * y

        def evaluate(x: np.ndarray, y: np.ndarray, z: np.ndarray, maxn_params=5) -> float:
            from scipy.optimize import minimize

            def loss(params):
                z_pred = equation(x, y, params)
                return np.mean((z_pred - z) ** 2)

            result = minimize(lambda p: loss(p), [1.0] * maxn_params, method="BFGS")
            loss_val = result.fun
            return -loss_val if np.isfinite(loss_val) else float('-inf')

        def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
            """Seed: simple additive model."""
            return params[0] * x + params[1] * y

        est = LLMSR(
            prompt="Find the mathematical function skeleton that best fits the data. "
                   "Return only the def equation(...) function with arguments (x, y, params).",
            eval_program=evaluate,
            seed_program=equation,
            namespace={"np": np},
            n_islands=3,
            n_iter=5,
            programs_per_prompt=1,
            save_path=None,
        )

        data = {"x": x, "y": y, "z": z}
        est.fit(data)

        assert est.best_model is not None
        assert "def equation" in est.best_model.program
