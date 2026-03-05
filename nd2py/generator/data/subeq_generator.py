import random
import numpy as np
from ...core.symbols import Variable
from .gmm_generator import GMMGenerator


class SubeqGenerator(GMMGenerator):
    """
    生成高斯随机分布的 z in R^{K} 以及 D 个随机方程 f_d。以 X_d=f_d(z) 计算 y=f(X_d')。
    其中 X_d' 是将每个 X_d 分别归一化到 N(0, 1) 的结果。
    X_d 中可能会有离群值，这会影响 X_d' 的分布范围，因此归一化之前先去除 threshold (%) 之外的值。
    """
    def __init__(self, eq_generator, max_value=np.inf, max_var=5, normalize_X=False):
        self.eq_generator = eq_generator
        self.max_value = max_value
        self.max_var = max_var
        self.normalize_X = normalize_X
    
    def generate_data(self, num, eqtree):
        variables = sorted(set(x.name for x in eqtree.preorder() if isinstance(x, Variable)))
        D = len(variables)
        K = random.randint(1, self.max_var)
        f = [self.eq_generator.generate_eqtree(n_var=random.randint(1, K)) for _ in range(D)]
        X_list, Y_list = [], []
        for _ in range(5):
            Z = self.GMM(num, K)
            Z = (Z - np.mean(Z, axis=0, keepdims=True)) / (np.std(Z, axis=0, keepdims=True) + 1e-6)
            Z = {f'x_{k+1}':Z[:, k] for k in range(K)}
            X = np.stack([f_d.eval(**Z) for f_d in f], axis=-1)
            if self.normalize_X: X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-6)
            Y = eqtree.eval(**{var: X[:, idx] for idx, var in enumerate(variables)})
            invalid_idx = np.isnan(Y) | np.isinf(Y) | (np.abs(Y) > self.max_value) | np.isnan(X).any(axis=-1) | np.isinf(X).any(axis=-1) | (np.abs(X) > self.max_value).any(axis=-1)
            X_list.append(X[~invalid_idx, :])
            Y_list.append(Y[~invalid_idx])
            if sum(map(len, Y_list)) >= num: break
        else:
            return None, None
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        shuffle_idx = np.random.permutation(len(Y))
        return X[shuffle_idx[:num], :], Y[shuffle_idx[:num]]