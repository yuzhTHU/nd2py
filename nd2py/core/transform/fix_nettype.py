from typing import Literal
from ..symbols import *
from ..base_visitor import Visitor


class FixNetType(Visitor):
    def __call__(
        self,
        node: Symbol,
        nettype: NetType = "node",
        direction: Literal["bottom-up", "top-down"] = "top-down",
        edge_to_node=["remove_targ", "remove_sour", "add_aggr", "add_rgga"],
        node_to_edge=["remove_aggr", "remove_rgga", "add_targ", "add_sour"],
        edge_to_scalar=["remove_sour", "remove_targ", "add_readout"],
        node_to_scalar=["remove_aggr", "remove_rgga", "add_readout"],
        scalar_to_node=["keep"],
        scalar_to_edge=["keep"],
        _outside=True,
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
        try:
            y = super().__call__(
                node,
                nettype=nettype,
                direction=direction,
                edge_to_node=edge_to_node,
                node_to_edge=node_to_edge,
                edge_to_scalar=edge_to_scalar,
                node_to_scalar=node_to_scalar,
                scalar_to_node=scalar_to_node,
                scalar_to_edge=scalar_to_edge,
                _outside=False,
            )
            if _outside and y.nettype != nettype:
                y = self.fix_nettype(
                    y,
                    nettype=nettype,
                    edge_to_node=edge_to_node,
                    node_to_edge=node_to_edge,
                    edge_to_scalar=edge_to_scalar,
                    node_to_scalar=node_to_scalar,
                    scalar_to_node=scalar_to_node,
                    scalar_to_edge=scalar_to_edge,
                )
            return y
        except Exception as e:
            raise ValueError(f"Error in {type(self).__name__}({node}): {e}") from e

    def generic_visit(self, node, *args, **kwargs):
        """
        direction = 'top-down': 每个 node 的 nettype 由 kwargs['nettype'] 决定。
        direction = 'bottom-up': 每个 node 的 nettype 由其 operands 决定。只保证每个 node 运算不会出错即可，不需要对 kwargs['nettype'] 负责
        """
        if node.n_operands != 1:
            raise NotImplementedError(f"visit_{type(node).__name__} is not implemented")

        if kwargs["direction"] == "top-down":
            x = self(node.operands[0], *args, **kwargs)
            return node.__class__(x, nettype=kwargs["nettype"])
        elif kwargs["direction"] == "bottom-up":
            x = self(node.operands[0], *args, **kwargs)
            return node.__class__(x, nettype=x.nettype)
        else:
            raise ValueError(f"Unsupported direction: {kwargs['direction']}")

    def visit_Number(self, node: Number, *args, **kwargs):
        if kwargs["direction"] == "top-down":
            return node.__class__(node.value, nettype=kwargs["nettype"])
        elif kwargs["direction"] == "bottom-up":
            return node
        else:
            raise ValueError(f"Unsupported direction: {kwargs['direction']}")

    def visit_Variable(self, node: Variable, *args, **kwargs):
        if kwargs["direction"] == "top-down":
            return self.fix_nettype(node, *args, **kwargs)
        elif kwargs["direction"] == "bottom-up":
            return node
        else:
            raise ValueError(f"Unsupported direction: {kwargs['direction']}")

    def visit_BinaryOp(self, node, *args, **kwargs):
        if kwargs["direction"] == "top-down":
            x1 = self(node.operands[0], *args, **kwargs)
            x2 = self(node.operands[1], *args, **kwargs)
            return node.__class__(x1, x2, nettype=kwargs["nettype"])
        elif kwargs["direction"] == "bottom-up":
            x1 = self(node.operands[0], *args, **kwargs)
            x2 = self(node.operands[1], *args, **kwargs)
            if x1.nettype == x2.nettype:
                return node.__class__(x1, x2, nettype=x1.nettype)
            elif {x1.nettype, x2.nettype} == {"node", "edge"}:
                x1 = self.fix_nettype(x1, *args, **kwargs)
                x2 = self.fix_nettype(x2, *args, **kwargs)
                return node.__class__(x1, x2, nettype=kwargs["nettype"])
            elif {x1.nettype, x2.nettype} == {"edge", "scalar"}:
                return node.__class__(x1, x2, nettype="edge")
            elif {x1.nettype, x2.nettype} == {"node", "scalar"}:
                return node.__class__(x1, x2, nettype="node")
            else:
                raise ValueError(
                    f"Unsupported nettype combination: {x1.nettype}, {x2.nettype}"
                )
        else:
            raise ValueError(f"Unsupported direction: {kwargs['direction']}")

    visit_Add = visit_BinaryOp
    visit_Sub = visit_BinaryOp
    visit_Mul = visit_BinaryOp
    visit_Div = visit_BinaryOp
    visit_Pow = visit_BinaryOp
    visit_Max = visit_BinaryOp
    visit_Min = visit_BinaryOp

    def visit_Aggr(self, node, *args, **kwargs):
        if kwargs["direction"] == "top-down":
            x = self(node.operands[0], *args, **(kwargs | {"nettype": "edge"}))
            y = node.__class__(x, nettype="node")
            y = self.fix_nettype(y, *args, **kwargs)
            return y
        elif kwargs["direction"] == "bottom-up":
            x = self(node.operands[0], *args, **(kwargs | {"nettype": "edge"}))
            if x.nettype == "node":
                x = self.fix_nettype(x, *args, **(kwargs | {"nettype": "edge"}))
            y = node.__class__(x, nettype="node")
            return y
        else:
            raise ValueError(f"Unsupported direction: {kwargs['direction']}")

    visit_Rgga = visit_Aggr

    def visit_Sour(self, node, *args, **kwargs):
        if kwargs["direction"] == "top-down":
            x = self(node.operands[0], *args, **(kwargs | {"nettype": "node"}))
            y = node.__class__(x, nettype="edge")
            y = self.fix_nettype(y, *args, **kwargs)
            return y
        elif kwargs["direction"] == "bottom-up":
            x = self(node.operands[0], *args, **(kwargs | {"nettype": "node"}))
            if x.nettype == "edge":
                x = self.fix_nettype(x, *args, **(kwargs | {"nettype": "node"}))
            y = node.__class__(x, nettype="edge")
            return y
        else:
            raise ValueError(f"Unsupported direction: {kwargs['direction']}")

    visit_Targ = visit_Sour

    def visit_Readout(self, node, *args, **kwargs):
        if kwargs["direction"] == "top-down":
            x = self(node.operands[0], *args, **(kwargs | {"nettype": "node"}))
            y = node.__class__(x, nettype="scalar")
            y = self.fix_nettype(y, *args, **kwargs)
            return y
        elif kwargs["direction"] == "bottom-up":
            x = self(node.operands[0], *args, **(kwargs | {"nettype": "node"}))
            if x.nettype == "scalar":
                return x
            y = node.__class__(x, nettype="scalar")
            return y
        else:
            raise ValueError(f"Unsupported direction: {kwargs['direction']}")

    def fix_nettype(self, node, *args, **kwargs):
        nettype = kwargs["nettype"]
        if node.nettype == nettype:
            return node
        elif node.nettype == "edge" and nettype == "node":
            return self.edge_to_node(node, *args, **kwargs)
        elif node.nettype == "edge" and nettype == "scalar":
            return self.edge_to_scalar(node, *args, **kwargs)
        elif node.nettype == "node" and nettype == "edge":
            return self.node_to_edge(node, *args, **kwargs)
        elif node.nettype == "node" and nettype == "scalar":
            return self.node_to_scalar(node, *args, **kwargs)
        elif node.nettype == "scalar" and nettype == "node":
            return self.scalar_to_node(node, *args, **kwargs)
        elif node.nettype == "scalar" and nettype == "edge":
            return self.scalar_to_edge(node, *args, **kwargs)
        else:
            raise ValueError(
                f"Unsupported nettype conversion from {node.nettype} to {nettype}"
            )

    def edge_to_node(self, node, *args, **kwargs):
        edge_to_node = kwargs["edge_to_node"]
        for method in edge_to_node:
            if method == "remove_targ" and isinstance(node, Targ):
                return node.operands[0]
            elif method == "remove_sour" and isinstance(node, Sour):
                return node.operands[0]
            elif method == "add_aggr":
                return Aggr(node)
            elif method == "add_rgga":
                return Rgga(node)
        raise ValueError(f"No valid edge to node conversion method found for {node}")

    def node_to_edge(self, node, *args, **kwargs):
        node_to_edge = kwargs["node_to_edge"]
        for method in node_to_edge:
            if method == "remove_aggr" and isinstance(node, Aggr):
                return node.operands[0]
            elif method == "remove_rgga" and isinstance(node, Rgga):
                return node.operands[0]
            elif method == "add_targ":
                return Targ(node)
            elif method == "add_sour":
                return Sour(node)
        raise ValueError(f"No valid node to edge conversion method found for {node}")

    def edge_to_scalar(self, node, *args, **kwargs):
        edge_to_scalar = kwargs["edge_to_scalar"]
        for method in edge_to_scalar:
            if (
                method == "remove_sour"
                and isinstance(node, Sour)
                and node.operands[0].nettype == "scalar"
            ):
                return node.operands[0]
            elif (
                method == "remove_targ"
                and isinstance(node, Targ)
                and node.operands[0].nettype == "scalar"
            ):
                return node.operands[0]
            elif method == "add_readout":
                return Readout(node)
        raise ValueError(f"No valid edge to scalar conversion method found for {node}")

    def node_to_scalar(self, node, *args, **kwargs):
        node_to_scalar = kwargs["node_to_scalar"]
        for method in node_to_scalar:
            if (
                method == "remove_aggr"
                and isinstance(node, Aggr)
                and node.operands[0].nettype == "scalar"
            ):
                return node.operands[0]
            elif (
                method == "remove_rgga"
                and isinstance(node, Rgga)
                and node.operands[0].nettype == "scalar"
            ):
                return node.operands[0]
            elif method == "add_readout":
                return Readout(node)
        raise ValueError(f"No valid node to scalar conversion method found for {node}")

    def scalar_to_node(self, node, *args, **kwargs):
        return node
        # raise NotImplementedError(
        #     f"scalar_to_node not implemented for {type(node).__name__}"
        # )

    def scalar_to_edge(self, node, *args, **kwargs):
        return node
        # raise NotImplementedError(
        #     f"scalar_to_edge not implemented for {type(node).__name__}"
        # )
