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
        """
        Args:
        - raw:bool=False, whether to return the raw format
        - number_format:str='', can be '0.2f'
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        - latex:bool=False, whether to return the latex format
        - skeleton:bool=False, whether to ignore the concrete values of Number
        """
        from .printer.string_printer import StringPrinter

        return StringPrinter()(
            self,
            raw=raw,
            latex=latex,
            number_format=number_format,
            omit_mul_sign=omit_mul_sign,
            skeleton=skeleton,
        )

    def to_tree(self, number_format="", flat=False, skeleton=False) -> str:
        """
        Args:
        - number_format:str='', can be '0.2f'
        - flat:bool=False, whether to flat the Add and Mul
        - omit_mul_sign:bool=False, whether to omit the multiplication sign
        """
        from .printer.tree_printer import TreePrinter

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
        """
        Args:
        - vars: a dictionary of variable names and their values
        - edge_list: edges (i,j) in the graph
            i and j are the indices of the nodes (starting from 0)
        - num_nodes: the number of nodes in the graph
            if not provided, it will be inferred from edge_list
        - use_eps: a small value to avoid division by zero
        """
        from .calc.numpy_calc import NumpyCalc

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
        """
        Args:
        - vars: a dictionary of variable names and their values
        - edge_list: edges (i,j) in the graph
            i and j are the indices of the nodes (starting from 0)
        - num_nodes: the number of nodes in the graph
            if not provided, it will be inferred from edge_list
        - use_eps: a small value to avoid division by zero
        - device: cpu or cuda
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
        """Split the node by addition, returning a list of symbols.
        Args:
        - node: Symbol, the node to split
        - split_by_sub: bool, whether to split by Sub nodes
        - expand_mul: bool, whether to expand Mul nodes
        - expand_div: bool, whether to expand Div nodes
        - expand_aggr: bool, whether to expand Aggr nodes
        - expand_rgga: bool, whether to expand Rgga nodes
        - expand_sour: bool, whether to expand Sour nodes
        - expand_targ: bool, whether to expand Targ nodes
        - expand_readout: bool, whether to expand Readout (Readout(a + b) -> [Readout(a), Readout(b)])
        - remove_coefficients: bool, whether to remove coefficients from the symbols
        - merge_bias: bool, whether to merge bias terms
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
        """Split the node by multiplication, returning a list of symbols.
        Args:
        - node: Symbol, the node to split (a * b * c -> [a, b, c])
        - split_by_div: bool, whether to split by Div (a / b -> [a, b])
        - merge_coefficients: bool, whether to merge coefficients from the symbols
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
        """fix the nettype of symbols in an expression, useful in GP or LLMSR where equations are generated randomly and can have incorrect nettypes
        - node: the root symbol of the expression to fix
        - nettype: the nettype to set for the symbols, can be 'node', 'edge', or 'scalar'
        - direction: the direction of the fix, can be 'bottom-up' or 'top-down'
        - edge_to_node: list of operations to convert edge symbols to node symbols
        - node_to_edge: list of operations to convert node symbols to edge symbols
        - edge_to_scalar: list of operations to convert edge symbols to scalar symbols
        - node_to_scalar: list of operations to convert node symbols to scalar symbols
        - scalar_to_node: list of operations to convert scalar symbols to node symbols
        - scalar_to_edge: list of operations to convert scalar symbols to edge symbols
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
