# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from .numpy_calc import NumpyCalc
from ..base_visitor import Visitor, yield_nothing
if TYPE_CHECKING:
    from ..symbols import Symbol


class EICCalc(Visitor):
    """Estimate the number of effective decimal digits lost by a symbol tree, see 
    `Beyond Accuracy and Complexity: The Effective Information Criterion for Structurally Stable Symbolic Regression'

    The estimate recursively measures how much each subexpression amplifies
    relative input errors. Analytic derivatives are used for common
    element-wise operators. Other operators are evaluated by
    :class:`NumpyCalc` and differentiated with a central finite difference in
    :meth:`generic_visit`.
    """
    def __call__(
        self,
        node: Symbol,
        vars: dict = {},
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        use_eps: float = 0.0,
        perturbation: float = 1e-6,
        return_exceptions: bool = False,
        exception_threshold: float = 1.0,
    ):
        if perturbation <= 0:
            raise ValueError("perturbation must be positive")
        if num_nodes is None and edge_list is not None:
            nodes = np.unique(np.asarray(edge_list).reshape(-1))
            num_nodes = int(np.max(nodes)) + 1

        exceptions = [] if return_exceptions else None
        _, _, eic = super().__call__(
            node,
            vars=vars,
            edge_list=edge_list,
            num_nodes=num_nodes,
            use_eps=use_eps,
            perturbation=perturbation,
            exceptions=exceptions,
            exception_threshold=exception_threshold,
        )
        if return_exceptions:
            return eic, exceptions
        return eic

    def visit_Empty(self, node, *args, **kwargs):
        raise ValueError(
            f"Incomplete expression with Empty node is not allowed to evaluate: {node}"
        )
        yield from yield_nothing()

    def visit_Number(self, node, *args, **kwargs):
        yield from yield_nothing()
        value = np.asarray(node.value)
        return np.ones_like(value, dtype=float), value, 0.0

    def visit_Variable(self, node, *args, **kwargs):
        yield from yield_nothing()
        value = np.asarray(kwargs["vars"][node.name])
        return np.ones_like(value, dtype=float), value, 0.0

    def visit_GroupedParameter(self, node, *args, **kwargs):
        """Treat a bound grouped parameter as an input leaf."""
        yield from yield_nothing()
        value = NumpyCalc()(
            node,
            vars=kwargs["vars"],
            edge_list=kwargs["edge_list"],
            num_nodes=kwargs["num_nodes"],
            use_eps=kwargs["use_eps"],
        )
        return np.ones_like(value, dtype=float), value, 0.0

    def generic_visit(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, None, args, kwargs))

    def _visit_operation(self, node, derivative, args, kwargs):
        children = []
        for operand in node.operands:
            children.append((yield (operand, args, kwargs)))

        child_s2 = [child[0] for child in children]
        child_values = [child[1] for child in children]
        child_eics = [child[2] for child in children]
        value = self._evaluate_operator(node, child_values, kwargs)

        if derivative is None:
            partials = self._finite_difference_partials(node, child_values, kwargs)
        else:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                partials = derivative(*child_values)
            if not isinstance(partials, (tuple, list)):
                partials = (partials,)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s2 = np.ones_like(np.asarray(value), dtype=float)
            for child_value, partial, child_variance in zip(
                child_values, partials, child_s2
            ):
                numerator, denominator = np.broadcast_arrays(
                    np.asarray(child_value) * partial, value
                )
                kappa = np.zeros_like(numerator, dtype=float)
                nonzero_output = denominator != 0
                np.divide(
                    numerator,
                    denominator,
                    out=kappa,
                    where=nonzero_output,
                )
                # If both numerator and output are zero, this input branch has
                # no first-order effect and its relative sensitivity is zero.
                # A nonzero numerator at zero output indicates singular
                # relative sensitivity, as in exact cancellation.
                singular = ~nonzero_output & (numerator != 0)
                kappa[singular] = np.inf
                kappa[~nonzero_output & np.isnan(numerator)] = np.nan
                squared_kappa = np.square(kappa)
                # A branch with exactly zero relative sensitivity contributes
                # no propagated variance, even if the child amplification is
                # infinite. Handle this limit explicitly to avoid 0 * inf.
                contribution = np.zeros_like(
                    np.broadcast_arrays(squared_kappa, child_variance)[0],
                    dtype=float,
                )
                np.multiply(
                    squared_kappa,
                    child_variance,
                    out=contribution,
                    where=squared_kappa != 0,
                )
                s2 = s2 + contribution

        mean_s2 = float(np.mean(s2))
        local_eic = 0.5 * np.log10(mean_s2)
        eic = float(max([0.0, local_eic, *child_eics]))
        self._record_high_eic(node, local_eic, kwargs)
        return s2, value, eic

    @staticmethod
    def _evaluate_operator(node, values, kwargs):
        method = getattr(NumpyCalc, "visit_" + type(node).__name__)
        operation = getattr(method, "__wrapped__", None)
        if operation is None:
            raise NotImplementedError(
                f"NumpyCalc.visit_{type(node).__name__} cannot be evaluated as an operator"
            )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return operation(NumpyCalc(), node, *values, **kwargs)

    def _finite_difference_partials(self, node, values, kwargs):
        relative_step = kwargs["perturbation"]
        partials = []
        for index, value in enumerate(values):
            value = np.asarray(value)
            if value.dtype.kind not in "biufc":
                raise TypeError(
                    f"Cannot numerically differentiate {node} with respect to "
                    f"non-numeric operand {index}"
                )
            step = relative_step * np.maximum(1.0, np.abs(value))
            plus = list(values)
            minus = list(values)
            plus[index] = value + step
            minus[index] = value - step
            y_plus = self._evaluate_operator(node, plus, kwargs)
            y_minus = self._evaluate_operator(node, minus, kwargs)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                partials.append((y_plus - y_minus) / (2.0 * step))
        return partials

    @staticmethod
    def _record_high_eic(node, local_eic, kwargs):
        exceptions = kwargs.get("exceptions")
        if exceptions is not None and (
            not np.isfinite(local_eic) or local_eic >= kwargs["exception_threshold"]
        ):
            exceptions.append(
                f"The subexpression {node} exhibits elevated local numerical "
                f"sensitivity, corresponding to an estimated loss of "
                f"{local_eic:.1f} decimal digits of precision."
            )

    def visit_Add(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x, y: (1, 1), args, kwargs))

    def visit_Sub(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x, y: (1, -1), args, kwargs))

    def visit_Mul(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x, y: (y, x), args, kwargs))

    def visit_Div(self, node, *args, **kwargs):
        def derivative(x, y):
            eps = kwargs["use_eps"]
            denominator = y + eps * (y == 0)
            return 1 / denominator, -x / denominator**2
        return (yield from self._visit_operation(node, derivative, args, kwargs))

    def visit_Pow(self, node, *args, **kwargs):
        def derivative(x, y):
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return y * x ** (y - 1), x**y * np.log(x)
        return (yield from self._visit_operation(node, derivative, args, kwargs))

    def visit_Identity(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: 1, args, kwargs))

    def visit_Sin(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: np.cos(x), args, kwargs))

    def visit_Cos(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: -np.sin(x), args, kwargs))

    def visit_Exp(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: np.exp(x), args, kwargs))

    def visit_Log(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: 1 / x, args, kwargs))

    def visit_Abs(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: np.sign(x), args, kwargs))

    def visit_Neg(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: -1, args, kwargs))

    def visit_Inv(self, node, *args, **kwargs):
        def derivative(x):
            eps = kwargs["use_eps"]
            denominator = x + eps * (x == 0)
            return -1 / denominator**2
        return (yield from self._visit_operation(node, derivative, args, kwargs))

    def visit_Sqrt(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: 0.5 / np.sqrt(x), args, kwargs))

    def visit_Pow2(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: 2 * x, args, kwargs))

    def visit_Pow3(self, node, *args, **kwargs):
        return (yield from self._visit_operation(node, lambda x: 3 * x**2, args, kwargs))

# \begin{algorithm}[tb]
#     \caption{CalculateEIC (Recursively calculate partial relative condition number)}
#     \label{alg:calculate_eic}
#     \begin{algorithmic}
#     \STATE {\bfseries Input:} Formula $f$ and data distribution $\mathcal{D}$ with $N$ samples.
#     \STATE {\bfseries Output:} Variance amplification factor $s(x)$ and output value $y(x)$ at root node, as well as EIC value of $f$.
#     \STATE
#     \STATE $k \gets f.\text{root}$ \quad \% Start from the root node
#     \IF{$k$ is a Leaf Node (Variable or Constant)}
#         \STATE $y_k(x) \gets \text{Evaluate}(k, \mathcal{D})$ \quad \% Get values of $k$ in $\mathcal{D}$
#         \STATE $s_k^2(x) \gets 1$
#         \STATE $\bar{s}_k^2 \gets \mathbb{E}_{x\sim\mathcal{D}} [s_k^2(x)]$ \quad \% $\bar{s}_k=1$ for leaf nodes
#         \STATE $\text{EIC}_k \gets \log_{10} \bar{s}_k$ \quad \% $\text{EIC}=0$ for leaf nodes
#         \STATE {\bfseries return} $(s_k(x), y_k(x), \text{EIC}_k)$
#     \ELSE
#         \STATE $\text{EIC}_k \gets 0$
#         \STATE $e_k \gets k.\text{operator}$
#         \FOR{each child $i \in \mathcal{C}[k]$}
#             \STATE $f' \gets \text{Subtree rooted at } i$
#             \STATE $(s_i(x), y_i(x), \text{EIC}_i) \gets \text{CalculateEIC}(f', \mathcal{D})$
#             \STATE $\text{EIC}_k \gets \max\{\text{EIC}_k, \text{EIC}_i\}$
#         \ENDFOR
#         \STATE $y_k(x) \gets e_k(\{y_{i}\}_{i\in\mathcal{C}[k]})$
#         \FOR{each child $i \in \mathcal{C}[k]$}
#             \STATE $\kappa_{k, i}(x) \gets \frac{y_i(x)}{y_k(x)} \frac{\partial e_{k}}{\partial y_i} |_{x\sim\mathcal{D}}$
#         \ENDFOR
#         \STATE $s_k^2(x) \gets 1 + \sum_{i \in \mathcal{C}[k]} \kappa_{k,i}^2 (x) s_i^2(x)$
#         \STATE $\bar{s}_k^2 \gets \mathbb{E}_{x\sim\mathcal{D}} [s_k^2(x)]$
#         \STATE $\text{EIC}_k \gets \max\{\text{EIC}_k, \log_{10} \bar{s}_k\}$
#         \STATE {\bfseries return} $(s_k(x), y_k(x), \text{EIC}_k)$
#     \ENDIF
#     \end{algorithmic}
# \end{algorithm}
