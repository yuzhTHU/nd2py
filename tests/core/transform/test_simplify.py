import pytest
import numpy as np
import nd2py as nd

np.random.seed(42)

x, y, z = nd.variables('x y z', nettype='scalar')
n = nd.Variable('n', nettype='node')
e = nd.Variable('e', nettype='edge')
s = nd.Variable('s', nettype='scalar')

@pytest.mark.parametrize("node,flags,expected", [
# ==========================================
        # 1. 常量树折叠 (transform_constant_subtree)
        # ==========================================
        # 默认情况下纯数字的树应该被计算
        (nd.Number(2) + nd.Number(3), {}, nd.Number(5)),
        # 当关闭 transform_constant_subtree 时，应当保留原样
        # (nd.Number(2) + nd.Number(3), {"transform_constant_subtree": False}, nd.Number(2) + nd.Number(3)),

        # ==========================================
        # 2. Add / Sub 的零值处理与负号提取 (visit_Add, visit_Sub)
        # ==========================================
        (x + 0, {}, x),
        (x - 0, {}, x),
        (0 - x, {}, -x),
        # 多个减号或负数提取：x - y - z 会被分组重组为 x - (y + z)
        (x - y - z, {}, x - (y + z)),
        (-x - y, {}, -(x + y)),

        # ==========================================
        # 3. Mul / Div 的重组化简 (visit_Mul, visit_Div)
        # ==========================================
        # 多个除法：x / y / z 会被组合分母重组为 x / (y * z)
        (x / y / z, {}, x / (y * z)),
        # 除以多个倒数项
        (x / nd.Inv(y), {}, x * y), 
        
        # ==========================================
        # 4. Neg (负号) 运算的深度穿透化简 (visit_Neg)
        # ==========================================
        # 负负得正
        (-(-x), {}, x),
        # -(x - y) -> y - x
        (-(x - y), {}, y - x),
        # 将负号传递给乘除法中的常数项
        (-(nd.Number(2) * x), {}, nd.Number(-2) * x),
        (-(nd.Number(3) / x), {}, nd.Number(-3) / x),

        # ==========================================
        # 5. Inv (倒数) 运算的深度穿透化简 (visit_Inv)
        # ==========================================
        # 常量的倒数直接计算
        (nd.Inv(nd.Number(4)), {}, nd.Number(0.25)),
        # nd.Inv(x / y) -> y / x
        (nd.Inv(x / y), {}, y / x),

        # ==========================================
        # 6. Aggr (聚合) 特殊提取规则 (visit_Aggr)
        # ==========================================
        # 标量的聚合应该提取为 Aggr(1) * scalar
        (nd.aggr(s), {}, nd.aggr(1) * s),

        # ==========================================
        # 7. Readout 开关测试 (remove_useless_readout)
        # ==========================================
        # 默认会移除 Readout(scalar)，关闭该 flag 后应保留
        (nd.readout(s), {"remove_useless_readout": False}, nd.readout(s)),

        # ==========================================
        # 8. 其他嵌套一元函数的消除 (remove_nested_xxx)
        # ==========================================
        # 测试化简器能否直接剥离最外层的相同一元运算符
        (nd.log(nd.log(x) + 1), {"remove_nested_log": True}, 1 + nd.log(x)),
        (nd.sqrt(nd.sqrt(x) * y), {"remove_nested_sqrt": True}, nd.sqrt(x) * y),
        (nd.tanh(nd.tanh(x) / 2), {"remove_nested_tanh": True}, 0.5 * nd.tanh(x)),
        (nd.cos(nd.cos(x)), {"remove_nested_cos": True}, nd.cos(x)),
])
def test_simplify(node: nd.Symbol, flags: dict, expected: nd.Symbol):
    """
    Test the simplify function with various nodes and flags.
    """
    simplified = node.simplify(**flags)
    assert str(simplified) == str(expected), f"Expected {expected}, got {simplified}"
