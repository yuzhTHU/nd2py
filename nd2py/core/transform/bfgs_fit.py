# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from ..symbols import Symbol, Number, GroupedParameter
from .fold_constant import FoldConstant


def collect_numbers(expression):
    numbers = []
    for op in expression.iter_preorder():
        if isinstance(op, (Number, GroupedParameter)) and op.fitable:
            numbers.append(op)
    return numbers


class BFGSFit(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        expression: Symbol,
        edge_list=None,
        num_nodes=None,
        use_eps=1e-8,
        method="BFGS",
        tol=1e-6,
        options=None,
        fold_constant=False,
    ):
        self.expression = expression
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.use_eps = use_eps
        self.method = method
        self.tol = tol
        self.options = options if options is not None else {}
        self.fold_constant = fold_constant

    def fit(self, X, y=None):
        if y is None:
            y = 0

        for parameter in self.expression.iter_preorder():
            if isinstance(parameter, GroupedParameter):
                parameter.bind(parameter.by.eval(vars=X))

        if self.fold_constant:
            fold = FoldConstant(fold_fitable=True, fold_constant=False)
            self.expression = fold(self.expression, vars={}, edge_list=self.edge_list, num_nodes=self.num_nodes)
            fold = FoldConstant(fold_fitable=False, fold_constant=True)
            expression = fold(self.expression, vars=X, edge_list=self.edge_list, num_nodes=self.num_nodes)
        else:
            expression = self.expression

        # 1. 收集需要优化的 Number 节点
        numbers = collect_numbers(expression)
        target_numbers = collect_numbers(self.expression)
        if len(numbers) == 0:
            self.n_iter_ = 0
            self.loss_ = None
            self.success_ = True
            self.message_ = "No parameters to optimize."
            return self

        # 2. 打平成参数向量
        init_vals = np.concatenate(
            [np.asarray(n.value, dtype=float).reshape(-1) for n in numbers]
        )
        split = np.cumsum([0] + [np.size(n.value) for n in numbers])

        # 3. 定义 loss
        def loss_fn(params):
            for n, i, j in zip(numbers, split[:-1], split[1:]):
                n.value = params[i:j].reshape(np.shape(n.value))
            y_pred = expression.eval(
                vars=X,
                edge_list=self.edge_list,
                num_nodes=self.num_nodes,
                use_eps=self.use_eps,
            )
            with np.errstate(all="ignore"):
                loss = np.mean((y_pred - y) ** 2)
            return loss

        # 4. 调用 scipy.optimize.minimize
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
            res = minimize(
                loss_fn, init_vals, method=self.method, tol=self.tol, options=self.options
            )

        # 5. 把最优参数写回 Number 节点，保存结果
        if len(numbers) != len(target_numbers):
            raise RuntimeError("Fitable parameter count changed during constant folding")
        for source, target in zip(numbers, target_numbers):
            if type(source) is not type(target):
                raise RuntimeError("Fitable parameter order changed during constant folding")
        for source, target, i, j in zip(numbers, target_numbers, split[:-1], split[1:]):
            source.value = res.x[i:j].reshape(np.shape(source.value))
            target.value = np.asarray(source.value, dtype=float).copy()

        self.n_iter_ = res.nit
        self.loss_ = res.fun
        self.success_ = res.success
        self.message_ = res.message
        return self

    def predict(self, X):
        """
        用拟合好的 expression 去计算新的 X 上的输出。
        """
        y_pred = self.expression.eval(
            vars=X,
            edge_list=self.edge_list,
            num_nodes=self.num_nodes,
            use_eps=self.use_eps,
        )
        return y_pred
