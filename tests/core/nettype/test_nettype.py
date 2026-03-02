import pytest
import nd2py as nd

s = nd.Variable("s", nettype="scalar")
n = nd.Variable("n", nettype="node")
e = nd.Variable("e", nettype="edge")


@pytest.mark.parametrize(
    "node,expected",
    [
        (s, "scalar"),
        (n, "node"),
        (e, "edge"),
        (nd.aggr(s), "node"),
        (nd.aggr(e), "node"),
        (nd.aggr(s).operands[0], "scalar"),
        (nd.aggr(e).operands[0], "edge"),
        (nd.rgga(s), "node"),
        (nd.rgga(e), "node"),
        (nd.rgga(s).operands[0], "scalar"),
        (nd.rgga(e).operands[0], "edge"),
        (nd.sour(s), "edge"),
        (nd.sour(n), "edge"),
        (nd.sour(s).operands[0], "scalar"),
        (nd.sour(n).operands[0], "node"),
        (nd.targ(s), "edge"),
        (nd.targ(n), "edge"),
        (nd.targ(s).operands[0], "scalar"),
        (nd.targ(n).operands[0], "node"),
        ((s + n), "node"),
        ((s + n).operands[0], "scalar"),
        ((s + n).operands[1], "node"),
        ((s + e), "edge"),
        ((s + e).operands[0], "scalar"),
        ((s + e).operands[1], "edge"),
        ((s + s), "scalar"),
        ((n + n), "node"),
        ((e + e), "edge"),
    ],
)
def test_nettype(node, expected):
    assert node.nettype == expected


@pytest.mark.parametrize(
    "node,expected",
    [
        (lambda: nd.aggr(n), (ValueError)),
        (lambda: nd.rgga(n), (ValueError)),
        (lambda: nd.sour(e), (ValueError)),
        (lambda: nd.targ(e), (ValueError)),
        (lambda: n + e, (ValueError)),
        (lambda: nd.Aggr(s, nettype="edge"), (ValueError)),
        (lambda: nd.Aggr(s, nettype="scalar"), (ValueError)),
        (lambda: nd.Aggr(e, nettype="edge"), (ValueError)),
        (lambda: nd.Aggr(e, nettype="scalar"), (ValueError)),
    ],
)
def test_nettype_error(node, expected):
    with pytest.raises(expected):
        node()

# ==========================================
# 补充测试：动态网络类型推导与 Top-Down/Bottom-Up 传播
# ==========================================

def test_nettype_dynamic_inference():
    """测试变量在未指定 nettype 时的动态范围推导"""
    u = nd.Empty()
    # 初始状态，u 的类型未知
    assert u.nettype is None
    assert u.possible_nettypes == {"scalar", "node", "edge"}

    # Bottom-up 推导：aggr 算子要求其操作数只能是 scalar 或 edge
    expr = nd.aggr(u)
    assert expr.nettype == "node"
    assert u.nettype is None
    assert u.possible_nettypes == {"scalar", "edge"}

    # 手动施加硬约束：将其锚定为 edge
    u.nettype = "edge"
    assert u.nettype == "edge"
    assert u.possible_nettypes == {"edge"}


@pytest.mark.parametrize(
    "assigned_nettype,expected_u,expected_v,possible_uv",
    [
        # 加法操作如果结果被约束为 scalar，那么操作数必定都是 scalar
        ("scalar", "scalar", "scalar", {"scalar"}),
        # 加法操作如果结果被约束为 node，操作数可能是 (node, node) 或 (node, scalar) 等
        # 所以 u, v 无法确定单一类型，但范围会缩小到 {scalar, node}
        ("node", None, None, {"scalar", "node"}),
    ]
)
def test_nettype_top_down_propagation(assigned_nettype, expected_u, expected_v, possible_uv):
    """测试通过指定父节点类型，Top-Down 反向限制子节点范围"""
    u = nd.Empty()
    v = nd.Empty()
    expr = u + v
    
    # 在没有约束时，均不确定
    assert expr.nettype is None
    assert u.nettype is None
    
    # 指定根节点的类型，触发 Top-Down 类型推导
    expr.nettype = assigned_nettype
    
    assert u.nettype == expected_u
    assert v.nettype == expected_v
    assert u.possible_nettypes == possible_uv
    assert v.possible_nettypes == possible_uv

# ==========================================
# 补充测试：边界限制、报错与属性访问拦截
# ==========================================

def test_nettype_setters_and_conflicts():
    """测试对 possible_nettypes 的非法赋值拦截和赋值空集的报错"""
    u = nd.Empty()
    
    # 直接修改 possible_nettypes 应被拦截
    with pytest.raises(AttributeError, match="Cannot set possible_nettypes directly"):
        u.possible_nettypes = {"scalar"}

    # 分配空集合会被拦截
    with pytest.raises(ValueError, match="Possible nettypes became empty"):
        u.nettype = set()


def test_nettype_inference_conflict():
    """测试整树推导中如果出现不可调和的逻辑冲突，能否正确抛出 ValueError"""
    u = nd.Empty()
    # 构建一棵合法的树
    expr = nd.aggr(u)
    
    # 强行将 u 指定为 node。
    # 由于 aggr(node) 在 map_nettype 中是非法的，这应该触发 Type Inference Conflict 报错。
    with pytest.raises(ValueError, match="Type Inference Conflict"):
        u.nettype = "node"


def test_nettype_range_property():
    """测试类的 nettype_range 缓存机制 (get_nettype_range)"""
    # 这里通过实例化后获取 __class__ 来测试，确保不同算子的可能值域缓存正确
    s_temp = nd.Empty(nettype="scalar")
    
    assert nd.aggr(s_temp).__class__.nettype_range == {"node"}
    assert nd.sour(s_temp).__class__.nettype_range == {"edge"}
    
    # Add 的操作理论上可能产生任何类型 (scalar, node, edge)
    add_op = s_temp + s_temp
    assert add_op.__class__.nettype_range == {"scalar", "node", "edge"}