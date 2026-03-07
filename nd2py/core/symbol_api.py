# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import List, Tuple, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from .symbols import Symbol
    from .nettype import NetType
    try:
        import numpy as np
        import torch
    except ImportError:
        pass

class SymbolAPIMixin:
    """
    Symbol API Mixin.
    
    该类集中管理 Symbol 对象对用户暴露的所有核心功能接口 (Facade)。
    所有具体逻辑均由底层的 Visitor 类实现。
    
    作为 Symbol 的父类混入，从而保证 symbols.py 的整洁与高可维护性。
    """

    def to_str(
        self,
        raw=False,
        latex=False,
        number_format="",
        omit_mul_sign=False,
        skeleton=False,
    ) -> str:
        """Return a string representation of the symbol expression.

        This is a thin wrapper around :class:`StringPrinter` and can produce
        raw, LaTeX, or skeleton forms of the expression.

        Args:
            raw (bool, optional): If True, return the internal raw
                representation instead of a prettified one. Defaults to False.
            latex (bool, optional): If True, format the expression as LaTeX.
                Defaults to False.
            number_format (str, optional): Format specifier used to print
                numeric constants (for example ``"0.2f"``). Defaults to an
                empty string, which uses the default formatting.
            omit_mul_sign (bool, optional): If True, omit explicit
                multiplication signs (for example render ``ab`` instead of
                ``a*b``). Defaults to False.
            skeleton (bool, optional): If True, ignore concrete numeric values
                and keep only the symbolic structure of the expression.
                Defaults to False.

        Returns:
            str: String representation of the symbol expression.
        """
        from .converter import StringPrinter

        return StringPrinter()(
            self,
            raw=raw,
            latex=latex,
            number_format=number_format,
            omit_mul_sign=omit_mul_sign,
            skeleton=skeleton,
        )

    def to_tree(self, number_format="", flat=False, skeleton=False) -> str:
        """Return an ASCII tree representation of the expression.

        Args:
            number_format (str, optional): Format specifier used to print
                numeric constants (for example ``"0.2f"``). Defaults to an
                empty string, which uses the default formatting.
            flat (bool, optional): If True, flatten nested ``Add`` and
                ``Mul`` nodes into a single level. Defaults to False.
            skeleton (bool, optional): If True, ignore concrete numeric values
                and keep only the symbolic structure of the expression.
                Defaults to False.

        Returns:
            str: Multi-line string visualising the expression tree.
        """
        from .converter import TreePrinter

        return TreePrinter()(
            self, number_format=number_format, flat=flat, skeleton=skeleton
        )

    def eval(
        self,
        vars: dict = {},
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        use_eps: float = 0.0,
    ):
        """Evaluate the expression numerically using NumPy.

        Args:
            vars (dict, optional): Mapping from variable names to their
                numerical values. Values can be scalars or ``numpy.ndarray``
                objects. Defaults to an empty dictionary.
            edge_list (Tuple[List[int], List[int]], optional): Pair of
                integer lists ``(sources, targets)`` describing directed
                edges in a graph. Node indices start from 0. If provided,
                this is used to parameterise graph-related symbols.
            num_nodes (int, optional): Number of nodes in the underlying
                graph. If omitted, it may be inferred from ``edge_list`` when
                possible.
            use_eps (float, optional): Small positive value added in
                denominators or other potentially unstable operations to avoid
                division by zero and improve numerical stability. Defaults to
                0.0.

        Returns:
            numpy.ndarray | float: Numerical evaluation result of the
            expression, whose shape depends on the symbol and inputs.
        """
        from .calc import NumpyCalc

        return NumpyCalc()(
            self, vars=vars, edge_list=edge_list, num_nodes=num_nodes, use_eps=use_eps
        )

    def eval_torch(
        self,
        vars: dict = {},
        edge_list: Tuple[List[int], List[int]] = None,
        num_nodes: int = None,
        use_eps: float = 0.0,
        device: str = "cpu",
    ):
        """Evaluate the expression numerically using PyTorch.

        Args:
            vars (dict, optional): Mapping from variable names to their
                numerical values. Values can be scalars or ``torch.Tensor``
                objects. Defaults to an empty dictionary.
            edge_list (Tuple[List[int], List[int]], optional): Pair of
                integer lists ``(sources, targets)`` describing directed
                edges in a graph. Node indices start from 0. If provided,
                this is used to parameterise graph-related symbols.
            num_nodes (int, optional): Number of nodes in the underlying
                graph. If omitted, it may be inferred from ``edge_list`` when
                possible.
            use_eps (float, optional): Small positive value added in
                denominators or other potentially unstable operations to avoid
                division by zero and improve numerical stability. Defaults to
                0.0.
            device (str, optional): Target device on which tensors are
                allocated and computations are performed, such as ``"cpu"``
                or ``"cuda"``. Defaults to ``"cpu"``.

        Returns:
            torch.Tensor: Numerical evaluation result of the expression.
        """
        from .calc.torch_calc import TorchCalc

        return TorchCalc()(
            self,
            vars=vars,
            edge_list=edge_list,
            num_nodes=num_nodes,
            use_eps=use_eps,
            device=device,
        )

    def split_by_add(
        self,
        split_by_sub: bool = False,
        expand_mul: bool = False,
        expand_div: bool = False,
        expand_aggr: bool = False,
        expand_rgga: bool = False,
        expand_sour: bool = False,
        expand_targ: bool = False,
        expand_readout: bool = False,
        remove_coefficients: bool = False,
        merge_bias: bool = False,
    ) -> List["Symbol"]:
        """Split an additive expression into its additive terms.

        The current symbol is treated as the root. Depending on the flags,
        this can also expand multiplication, division, aggregation and
        readout nodes before splitting.

        Args:
            split_by_sub (bool, optional): If True, treat subtraction nodes
                as additions when splitting, so that ``a - b`` becomes
                ``[a, -b]``. Defaults to False.
            expand_mul (bool, optional): If True, expand ``Mul`` nodes before
                splitting. Defaults to False.
            expand_div (bool, optional): If True, expand ``Div`` nodes before
                splitting. Defaults to False.
            expand_aggr (bool, optional): If True, expand aggregation nodes
                (for example graph aggregators) before splitting. Defaults to
                False.
            expand_rgga (bool, optional): If True, expand RGGA-related nodes
                before splitting. Defaults to False.
            expand_sour (bool, optional): If True, expand source-related
                transformations before splitting. Defaults to False.
            expand_targ (bool, optional): If True, expand target-related
                transformations before splitting. Defaults to False.
            expand_readout (bool, optional): If True, push ``Readout`` inside
                additions, so that for example ``Readout(a + b)`` becomes
                ``[Readout(a), Readout(b)]``. Defaults to False.
            remove_coefficients (bool, optional): If True, drop scalar
                coefficients from the resulting symbols. Defaults to False.
            merge_bias (bool, optional): If True, merge additive bias terms
                into neighbouring symbols when appropriate. Defaults to False.

        Returns:
            List[Symbol]: List of symbols corresponding to each additive term.
        """
        from .transform.split_by_add import SplitByAdd

        return SplitByAdd()(
            self,
            split_by_sub=split_by_sub,
            expand_mul=expand_mul,
            expand_div=expand_div,
            expand_aggr=expand_aggr,
            expand_rgga=expand_rgga,
            expand_sour=expand_sour,
            expand_targ=expand_targ,
            expand_readout=expand_readout,
            remove_coefficients=remove_coefficients,
            merge_bias=merge_bias,
        )

    def split_by_mul(
        self,
        split_by_div: bool = False,
        merge_coefficients: bool = False,
    ) -> List["Symbol"]:
        """Split a multiplicative expression into its multiplicative factors.

        The current symbol is treated as the root. Depending on the flags,
        this can also split divisions and optionally merge coefficients.

        Args:
            split_by_div (bool, optional): If True, split divisions so that
                an expression like ``a / b`` is treated as having factors
                ``[a, b]``. Defaults to False.
            merge_coefficients (bool, optional): If True, merge scalar
                coefficients into a single factor instead of returning them as
                separate symbols. Defaults to False.

        Returns:
            List[Symbol]: List of symbols corresponding to each multiplicative
            factor.
        """
        from .transform.split_by_mul import SplitByMul

        return SplitByMul()(
            self,
            split_by_div=split_by_div,
            merge_coefficients=merge_coefficients,
        )

    def fix_nettype(
        self,
        nettype: NetType = "node",
        direction: Literal["bottom-up", "top-down"] = "top-down",
        edge_to_node=["remove_targ", "remove_sour", "add_aggr", "add_rgga"],
        node_to_edge=["remove_aggr", "remove_rgga", "add_targ", "add_sour"],
        edge_to_scalar=["remove_sour", "remove_targ", "add_readout"],
        node_to_scalar=["remove_aggr", "remove_rgga", "add_readout"],
        scalar_to_node=["keep"],
        scalar_to_edge=["keep"],
    ):
        """Normalize nettypes of all symbols in the expression.

        This is useful in GP or LLM-based symbolic regression where equations
        are generated automatically and may contain inconsistent nettype
        annotations.

        Args:
            nettype (NetType, optional): Target nettype of the root symbol.
                Typical values include ``"node"``, ``"edge"`` and
                ``"scalar"``. Defaults to ``"node"``.
            direction (Literal["bottom-up", "top-down"], optional): Direction
                in which the fix is propagated through the expression tree.
                Defaults to ``"top-down"``.
            edge_to_node (List[str], optional): Sequence of transformation
                rules applied when converting edge symbols to node symbols.
            node_to_edge (List[str], optional): Sequence of transformation
                rules applied when converting node symbols to edge symbols.
            edge_to_scalar (List[str], optional): Sequence of transformation
                rules applied when converting edge symbols to scalar symbols.
            node_to_scalar (List[str], optional): Sequence of transformation
                rules applied when converting node symbols to scalar symbols.
            scalar_to_node (List[str], optional): Sequence of transformation
                rules applied when converting scalar symbols to node symbols.
            scalar_to_edge (List[str], optional): Sequence of transformation
                rules applied when converting scalar symbols to edge symbols.

        Returns:
            Symbol: Root symbol of the expression with consistent nettypes.
        """
        from .transform.fix_nettype import FixNetType

        return FixNetType()(
            self,
            nettype=nettype,
            direction=direction,
            edge_to_node=edge_to_node,
            node_to_edge=node_to_edge,
            edge_to_scalar=edge_to_scalar,
            node_to_scalar=node_to_scalar,
            scalar_to_node=scalar_to_node,
            scalar_to_edge=scalar_to_edge,
        )

    def simplify(
        self,
        transform_constant_subtree: bool = True,
        remove_useless_readout: bool = True,
        remove_nested_sin: bool = False,
        remove_nested_cos: bool = False,
        remove_nested_tanh: bool = False,
        remove_nested_sigmoid: bool = False,
        remove_nested_sqrt: bool = False,
        remove_nested_sqrtabs: bool = False,
        remove_nested_exp: bool = False,
        remove_nested_log: bool = False,
        remove_nested_logabs: bool = False,
    ):
        """Apply algebraic simplifications to the expression.

        Each flag controls whether a specific family of simplification rules
        is enabled. By default constant subtrees are folded and useless
        readout nodes are removed.

        Args:
            transform_constant_subtree (bool, optional): If True, evaluate
                and replace constant-only subtrees with their numerical
                result. Defaults to True.
            remove_useless_readout (bool, optional): If True, eliminate
                redundant ``Readout`` nodes that do not affect the result.
                Defaults to True.
            remove_nested_sin (bool, optional): If True, simplify expressions
                containing nested sine functions when possible. Defaults to
                False.
            remove_nested_cos (bool, optional): If True, simplify expressions
                containing nested cosine functions when possible. Defaults to
                False.
            remove_nested_tanh (bool, optional): If True, simplify
                expressions containing nested hyperbolic tangent functions
                when possible. Defaults to False.
            remove_nested_sigmoid (bool, optional): If True, simplify
                expressions containing nested sigmoid functions when possible.
                Defaults to False.
            remove_nested_sqrt (bool, optional): If True, simplify nested
                square root expressions when possible. Defaults to False.
            remove_nested_sqrtabs (bool, optional): If True, simplify nested
                ``sqrtabs``-like expressions when possible. Defaults to False.
            remove_nested_exp (bool, optional): If True, simplify nested
                exponential expressions when possible. Defaults to False.
            remove_nested_log (bool, optional): If True, simplify nested
                logarithm expressions when possible. Defaults to False.
            remove_nested_logabs (bool, optional): If True, simplify nested
                ``logabs``-like expressions when possible. Defaults to False.

        Returns:
            Symbol: A simplified version of the original symbol expression.
        """
        from .transform.simplify import Simplify

        return Simplify()(
            self,
            transform_constant_subtree=transform_constant_subtree,
            remove_useless_readout=remove_useless_readout,
            remove_nested_sin=remove_nested_sin,
            remove_nested_cos=remove_nested_cos,
            remove_nested_tanh=remove_nested_tanh,
            remove_nested_sigmoid=remove_nested_sigmoid,
            remove_nested_sqrt=remove_nested_sqrt,
            remove_nested_sqrtabs=remove_nested_sqrtabs,
            remove_nested_exp=remove_nested_exp,
            remove_nested_log=remove_nested_log,
            remove_nested_logabs=remove_nested_logabs,
        )
