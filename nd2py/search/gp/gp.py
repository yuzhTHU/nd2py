from __future__ import annotations
import json
import time
import logging
import sklearn
import numpy as np
import pandas as pd
from numpy.random import RandomState, default_rng
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Generator, Optional, Literal, Set, TYPE_CHECKING
from ... import core as nd
from ...utils.timing import Timer, NamedTimer
from sklearn.base import BaseEstimator, RegressorMixin
from ...generator.eq.gplearn_generator import GPLearnGenerator
from ... import BFGSFit
if TYPE_CHECKING:
    from ...core import *


_logger = logging.getLogger(__name__)


class Individual:
    def __init__(self, eqtree: Symbol):
        assert isinstance(eqtree, nd.Identity), f"Sentinel Node (Identity) Missing! Got {type(eqtree)}"
        self.eqtree = eqtree
        self.complexity = None
        self.accuracy = None
        self.fitness = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Individual({self.eqtree.to_str(number_format=".2f")})'

    def copy(self) -> "Individual":
        copy = Individual(self.eqtree.copy())
        copy.complexity = self.complexity
        copy.accuracy = self.accuracy
        copy.fitness = self.fitness
        return copy


class GP(BaseEstimator, RegressorMixin):
    """
    Genetic Programming-based Symbolic Regression
    """

    def __init__(
        self,
        variables: List[Variable],
        binary: List[Symbol] = [nd.Add, nd.Sub, nd.Mul, nd.Div, nd.Max, nd.Min],
        unary: List[Symbol] = [nd.Sqrt, nd.Log, nd.Abs, nd.Neg, nd.Inv, nd.Sin, nd.Cos, nd.Tan],
        max_params: int = 2,
        elitism_k: int = 10,
        population_size: int = 1000,
        tournament_size: int = 20,
        p_crossover: float = 0.9,
        p_subtree_mutation: float = 0.01,
        p_hoist_mutation: float = 0.01,
        p_point_mutation: float = 0.01,
        p_point_replace: float = 0.05,
        const_range: Tuple[float, float] = (-1.0, 1.0),
        depth_range: Tuple[int, int] = (2, 6),
        full_prob: float = 0.5,
        nettype: Optional[Literal["node", "edge", "scalar"]] = "scalar",
        n_jobs: int = None,
        log_per_iter: int = float("inf"),
        log_per_sec: float = float("inf"),
        log_detailed_speed: bool = False,
        save_path: str = None,
        random_state: Optional[int] = None,
        n_iter=100,
        use_tqdm=False,
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        **kwargs,
    ):
        """Initialize a genetic-programming-based symbolic regression estimator.

        This configures the function set, search hyperparameters, logging
        behavior, and optional graph structure used for nettype-aware
        expressions.

        Args:
            variables (List[Variable]): List of input variables that can be
                used in generated expressions.
            binary (List[Symbol], optional): Binary operator symbols available
                to the GP (for example ``Add``, ``Sub``, ``Mul``). Defaults to
                a standard arithmetic and min/max set.
            unary (List[Symbol], optional): Unary operator symbols available
                to the GP (for example ``Sqrt``, ``Log``, ``Sin``). Defaults
                to a standard set of common functions.
            max_params (int, optional): Maximum number of numeric parameters
                (``Number`` nodes) allowed in an expression. Defaults to 2.
            elitism_k (int, optional): Number of top individuals carried over
                unchanged between generations. Defaults to 10.
            population_size (int, optional): Number of individuals in each
                generation. Defaults to 1000.
            tournament_size (int, optional): Number of individuals competing
                in each tournament during parent selection. Defaults to 20.
            p_crossover (float, optional): Probability of applying subtree
                crossover. Defaults to 0.9.
            p_subtree_mutation (float, optional): Probability of applying
                subtree mutation. Defaults to 0.01.
            p_hoist_mutation (float, optional): Probability of applying hoist
                mutation. Defaults to 0.01.
            p_point_mutation (float, optional): Probability of applying point
                mutation. Defaults to 0.01.
            p_point_replace (float, optional): Probability of replacing a
                node during point mutation. Defaults to 0.05.
            const_range (Tuple[float, float], optional): Range from which
                random constants are sampled. Defaults to ``(-1.0, 1.0)``.
            depth_range (Tuple[int, int], optional): Minimum and maximum tree
                depth for randomly generated expressions. Defaults to
                ``(2, 6)``.
            full_prob (float, optional): Probability of using the "full"
                method rather than "grow" when generating random trees.
                Defaults to 0.5.
            nettype (Optional[Literal["node", "edge", "scalar"]], optional):
                Nettype of the target expression, used when working with graph
                data. Defaults to ``"scalar"``.
            n_jobs (int, optional): Number of parallel jobs used for evolving
                the population. If ``None``, evolution is run in a single
                process. Defaults to ``None``.
            log_per_iter (float, optional): Log progress every ``log_per_iter``
                iterations; use ``float("inf")`` to disable iteration-based
                logging. Defaults to ``float("inf")``.
            log_per_sec (float, optional): Log progress every ``log_per_sec``
                seconds; use ``float("inf")`` to disable time-based logging.
                Defaults to ``float("inf")``.
            log_detailed_speed (bool, optional): If True, include detailed
                timing information for individual steps in logs. Defaults to
                False.
            save_path (str, optional): File path to which JSON lines of
                per-iteration records are appended. If ``None``, records are
                not written to disk. Defaults to ``None``.
            random_state (Optional[int], optional): Seed for the internal RNG
                to make runs reproducible. Defaults to ``None``.
            n_iter (int, optional): Maximum number of evolution iterations.
                Defaults to 100.
            use_tqdm (bool, optional): If True, wrap the main evolution loop
                with a ``tqdm`` progress bar. Defaults to False.
            edge_list (Tuple[List[int], List[int]], optional): Optional graph
                edge list ``(sources, targets)`` used when evaluating graph
                operators. If provided and ``num_nodes`` is ``None``, the
                number of nodes is inferred. Defaults to ``None``.
            num_nodes (int, optional): Number of nodes in the underlying
                graph. If ``None`` and ``edge_list`` is provided, it is
                inferred from the edges. Defaults to ``None``.
            **kwargs: Additional unused keyword arguments; a warning is logged
                if any are provided.

        Raises:
            AssertionError: If the sum of mutation and crossover probabilities
                exceeds 1.0.
        """
        if num_nodes is None and edge_list is not None:
            num_nodes = np.reshape(edge_list, (-1,)).max() + 1

        self.eqtree = None
        self.variables = variables
        self.binary = binary
        self.unary = unary
        self.max_params = max_params
        self.elitism_k = elitism_k
        self.random_state = random_state
        self._rng = default_rng(random_state)
        self.nettype = nettype

        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.const_range = const_range
        self.depth_range = depth_range
        self.full_prob = full_prob

        self.population_size = population_size
        self.tournament_size = tournament_size
        self.method_probs = {
            "crossover": p_crossover,  # replace a subtree of parents with a subtree of another parent
            "subtree-mutation": p_subtree_mutation,  # replace a subtree of parent with a random tree
            "hoist-mutation": p_hoist_mutation,  # select a random subtree of parent
            "point-mutation": p_point_mutation,  # replace a random node of parent with a symbol of same arity
        }
        self.p_point_replace = p_point_replace
        assert sum(self.method_probs.values()) <= 1
        self.method_probs["reproduction"] = 1 - sum(self.method_probs.values())

        self.generator = GPLearnGenerator(
            variables=self.variables,
            binary=self.binary,
            unary=self.unary,
            const_range=const_range,
            depth_range=depth_range,
            full_prob=full_prob,
            rng=self._rng,
            edge_list=edge_list,
            num_nodes=num_nodes,
        )

        self.n_jobs = n_jobs
        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.log_detailed_speed = log_detailed_speed
        self.records = []
        self.speed_timer = Timer()
        self.named_timer = NamedTimer()
        self.save_path = save_path
        self.n_iter = n_iter
        self.use_tqdm = use_tqdm
        self.edge_list = edge_list
        self.num_nodes = num_nodes

        if kwargs:
            _logger.warning(
                "Unknown args: %s", ", ".join(f"{k}={v}" for k, v in kwargs.items())
            )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray],
        y: np.ndarray | pd.Series,
    ):
        """Fit the GP model to training data by evolving expression trees.

        The input features can be provided as a NumPy array, a pandas
        ``DataFrame``, or a dictionary mapping variable names to arrays. The
        method runs the evolutionary loop, tracks the best individual, and
        stores its expression tree in ``self.eqtree``.

        Args:
            X (ndarray | DataFrame | Dict[str, ndarray]): Input features with
                shape ``(n_samples, n_dims)`` or an equivalent mapping from
                variable names to 1D arrays.
            y (ndarray | Series): Target values with shape ``(n_samples,)``.

        Returns:
            GP: The fitted estimator instance.

        Raises:
            ValueError: If ``X`` is of an unsupported type.
        """
        if isinstance(X, np.ndarray):
            X = {var.name: x for var, x in zip(self.variables, X[..., :])}
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for i, col in enumerate(X.columns)}
        elif isinstance(X, dict):
            X = {k: np.asarray(v) for k, v in X.items()}
        else:
            raise ValueError(f"Unknown type: {type(X)}")

        self.named_timer.clear(reset_last_add_time=True)
        self.speed_timer.clear(reset_last_add_time=True)
        self.start_time = time.time()
        population = self.init_population(X, y)
        for iter in tqdm(range(1, 1 + self.n_iter), disable=not self.use_tqdm):

            if self.n_jobs:
                population = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.evolve)(
                        population, X, y, self.population_size // self.n_jobs
                    )
                    for _ in range(self.n_jobs)
                )
                population = sum(population, [])
            else:
                population = self.evolve(population, X, y)

            self.speed_timer.add()

            best = max(population, key=lambda x: x.fitness).copy()
            self.eqtree = best.eqtree

            record = dict(
                iter=iter,
                time=time.time() - self.start_time,
                fitness=best.fitness,
                complexity=best.complexity,
                mse=best.accuracy,
                r2=float(1 - best.accuracy / y.var()),
                eqtree=str(best),
                population_size=len(population),
            )
            if (
                not iter % self.log_per_iter
                or self.named_timer.time > self.log_per_sec
            ):
                log = {}
                log["Iter"] = record["iter"]
                log["Fitness"] = record["fitness"]
                log["Complexity"] = record["complexity"]
                log["MSE"] = record["mse"]
                log["R2"] = record["r2"]
                log["Best equation"] = record["eqtree"]
                if self.log_detailed_speed:
                    log["Speed"] = str(self.speed_timer)
                    log["Time"] = str(self.named_timer)
                log["Population size"] = record["population_size"]
                _logger.info(
                    " | ".join(f"\033[4m{k}\033[0m: {v}" for k, v in log.items())
                )
                self.named_timer.clear()
                self.speed_timer.clear()

            record = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in record.items()
            }
            self.records.append(record)
            if self.save_path:
                with open(self.save_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            if best.accuracy < 1e-6:
                _logger.info(
                    f"Early stopping at iter {iter} with accuracy {best.accuracy} ({best})"
                )
                break
            self.named_timer.add("postprocess")
        return self

    def predict(
        self, X: np.ndarray | pd.DataFrame | Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Predict target values for ``X`` using the best evolved expression tree."""
        if self.eqtree is None:
            raise ValueError("Model not fitted yet")

        if isinstance(X, np.ndarray):
            X = {f"x_{i+1}": X[:, i] for i in range(X.shape[1])}
        elif isinstance(X, pd.DataFrame):
            X = {col: X[col].values for col in X.columns}
        elif isinstance(X, dict):
            pass
        else:
            raise ValueError(f"Unknown type: {type(X)}")
        return self.eqtree.eval(
            vars=X, edge_list=self.edge_list, num_nodes=self.num_nodes
        )

    def evolve(
        self,
        population: List[Individual],
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        children_size=None,
        elitism_k=None,
    ) -> List[Individual]:
        """Evolve a population for one generation and return the offspring."""
        if children_size is None:
            children_size = self.population_size
        if elitism_k is None:
            elitism_k = self.elitism_k
        assert (
            children_size > elitism_k
        ), f"children_size {children_size} must be greater than elitism_k {elitism_k}"

        children = []

        top_k = sorted(population, key=lambda x: x.fitness, reverse=True)[:elitism_k]
        children.extend(top_k)

        for parent in tqdm(
            self.tournament(population, children_size - elitism_k),
            disable=not self.use_tqdm,
        ):
            method = list(self.method_probs.keys())[
                np.searchsorted(
                    np.cumsum(list(self.method_probs.values())), self._rng.random()
                )
            ]
            if method == "crossover":
                donor = self.tournament(population, 1)[0]
                child = self.crossover(parent, donor)
            elif method == "subtree-mutation":
                child = self.subtree_mutation(parent)
            elif method == "hoist-mutation":
                child = self.hoist_mutation(parent)
            elif method == "point-mutation":
                child = self.point_mutation(parent)
            elif method == "reproduction":
                child = Individual(parent.eqtree.copy())
            else:
                raise ValueError(f"Unknown method: {method}")
            if (
                self._rng.random() < 0.1
                and 0
                < sum(type(op).__name__ == 'Number' for op in child.eqtree.iter_preorder())
                <= self.max_params
            ):
                bfgs_fit = BFGSFit(
                    child.eqtree, edge_list=self.edge_list, num_nodes=self.num_nodes
                )
                bfgs_fit.fit(X, y)
                child.eqtree = bfgs_fit.expression
            self.named_timer.add("evolve")
            self.set_fitness(child, X, y)
            children.append(child)
            self.named_timer.add("evaluate")
        return children

    def init_population(
        self, X: Dict[str, np.ndarray], y: np.ndarray
    ) -> List[Individual]:
        """Initialize the first population of individuals from the generator."""
        population = []
        for _ in range(self.population_size):
            eqtree = nd.Identity(empty:=nd.Empty(), nettype=self.nettype)
            random_tree = self.generator.sample(nettypes=eqtree.possible_nettypes, assign_root_nettypes=False)
            eqtree = eqtree.replace(empty, random_tree)
            individual = Individual(eqtree)
            self.set_fitness(individual, X, y)
            population.append(individual)
        return population

    def tournament(self, population: List[Individual], num) -> List[Individual]:
        """Select individuals from the population via tournament selection."""
        tournaments = self._rng.choice(
            population, size=(num, self.tournament_size), replace=True
        )
        winners = [
            max(tournament, key=lambda x: x.fitness) for tournament in tournaments
        ]
        return winners

    def crossover(self, parent: Individual, donor: Individual) -> Individual:
        """Create a child by replacing a subtree of ``parent`` with one from ``donor``."""
        child = parent.copy()
        subtree = self.get_random_subtree(child.eqtree.operands[0])
        child.eqtree = child.eqtree.replace(subtree, _subtree:=nd.Empty())
        donored_tree = self.get_random_subtree(donor.eqtree.operands[0], nettypes=_subtree.possible_nettypes)
        child.eqtree = child.eqtree.replace(_subtree, donored_tree)
        return child

    def subtree_mutation(self, parent: Individual) -> Individual:
        """Create a child by replacing a random subtree with a newly generated random tree."""
        child = parent.copy()
        subtree = self.get_random_subtree(child.eqtree.operands[0])
        child.eqtree = child.eqtree.replace(subtree, _subtree:=nd.Empty())
        random_tree = self.generator.sample(nettypes=_subtree.possible_nettypes, assign_root_nettypes=False)
        child.eqtree = child.eqtree.replace(_subtree, random_tree)
        return child

    def hoist_mutation(self, parent: Individual) -> Individual:
        """Create a child by hoisting a randomly chosen subtree to the root."""
        child = parent.copy()
        eqtree = child.eqtree.operands[0]
        subtree = self.get_random_subtree(eqtree, nettypes=eqtree.possible_nettypes)
        child.eqtree = child.eqtree.replace(eqtree, subtree)
        return child

    def point_mutation(self, parent: Individual) -> Individual:
        """Create a child by performing point mutation with random symbol replacements or insertions."""
        child = parent.copy()
        return child
        mutate_nodes = [
            node
            for node in child.eqtree.iter_postorder()  # Use postorder to ensure we mutate deeper nodes first
            if self._rng.random() < self.p_point_replace
        ]
        for node in mutate_nodes:
            if node.n_operands == 0:
                # if self._rng.integers(0, 1):
                sym = self.generator.generate_leaf(nettype=node.nettype)
                # else:
                #     sym = self._rng.choice(self.unary)(node.copy())
            elif node.n_operands == 1:
                child_types = [op.nettype for op in node.operands]
                nodes = [
                    sym
                    for sym in self.unary
                    if sym.map_nettype(*child_types) in node.possible_nettypes
                ]
                sym = self._rng.choice(nodes)(*node.operands)
            elif node.n_operands == 2:
                # if self._rng.integers(0, 1):
                child_types = [op.nettype for op in node.operands]
                nodes = [
                    sym
                    for sym in self.binary
                    if sym.map_nettype(*child_types) in node.possible_nettypes
                ]
                assert (
                    len(nodes) > 0
                ), f"No possible nettype for {node} with {child_types}"
                sym = self._rng.choice(nodes)(*node.operands)
                # else:
                #     sym = self._rng.choice(self.unary)(node.copy())
            else:
                raise ValueError(f"Unknown arity: {node.n_operands}")
            child.eqtree = child.eqtree.replace(node, sym)
        return child

    def set_fitness(self, individual: Individual, X: Dict[str, np.ndarray], y: np.ndarray):
        """Compute and assign complexity, accuracy, and fitness for an individual."""
        individual.complexity = len(individual.eqtree)
        self.named_timer.add("drop")
        y_pred = individual.eqtree.eval(
            vars=X, edge_list=self.edge_list, num_nodes=self.num_nodes
        )
        self.named_timer.add("evaluate.calc")
        with np.errstate(all="ignore"):
            individual.accuracy = float(np.mean((y_pred - y) ** 2))
        individual.fitness = float(
            0.999**individual.complexity / (1 + individual.accuracy / y.var())
        )
        if not np.isfinite(individual.fitness):
            individual.fitness = 0

    def get_random_subtree(
        self,
        individual: Individual | Symbol,
        nettypes: Set[Literal["node", "edge", "scalar"]] = None,
    ) -> Symbol:
        """Sample a subtree from an individual following the GPlearn/Koza (1992) node-selection strategy."""
        if isinstance(individual, Individual):
            individual = individual.eqtree
        if isinstance(nettypes, str):
            nettypes = {nettypes}
        if nettypes is None:
            nodes = [op for op in individual.iter_preorder()]
        else:
            nodes = []
            for op in individual.iter_preorder():
                if op.nettype in nettypes:
                    nodes.append(op)
                elif op.nettype == "edge" and "node" in nettypes:
                    if nd.Aggr in self.unary:
                        nodes.append(nd.Aggr(op))
                    if nd.Rgga in self.unary:
                        nodes.append(nd.Rgga(op))
                elif op.nettype == "node" and "edge" in nettypes:
                    if nd.Sour in self.unary:
                        nodes.append(nd.Sour(op))
                    if nd.Targ in self.unary:
                        nodes.append(nd.Targ(op))
        if len(nodes) == 0:
            return self.generator.sample(nettypes=nettypes, assign_root_nettypes=False)

        probs = np.array([0.9 if node.n_operands > 0 else 0.1 for node in nodes])
        subtree = nodes[
            np.searchsorted(np.cumsum(probs / probs.sum()), self._rng.random())
        ]
        return subtree
