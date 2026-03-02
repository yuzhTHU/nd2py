import pytest
import nd2py as nd

x = nd.Variable("x")
y = nd.Variable("y")
z = nd.Variable("z")

@pytest.mark.parametrize(
    "node,flags,expected",
    [
        # 基础乘法测试
        (x * y, {}, [x, y]),
        (x * y * z, {}, [x, y, z]),
        
        # 基础除法测试 (split_by_div)
        (x / y, {"split_by_div": True}, [x, nd.Inv(y)]), 
        
        # 测试 Number 的倒数与 fitable 传参问题 (需确保代码中已修复 fitable 赋值)
        (x / nd.Number(2.0, fitable=True), {"split_by_div": True}, [x, nd.Number(0.5, fitable=True)]),
        
        # 测试复杂的嵌套除法
        ((x / y) / z, {"split_by_div": True}, [x, nd.Inv(y), nd.Inv(z)]),
        
        # 测试 Inv 的反转逻辑
        (x / nd.Inv(y), {"split_by_div": True}, [x, y]),
        
        # 测试系数合并 (merge_coefficients)
        (nd.Number(2.0) * x * nd.Number(3.0) * y, {"merge_coefficients": True}, [nd.Number(6.0), x, y]),
        
        # 混合除法与系数合并
        (nd.Number(4.0) * x / nd.Number(2.0), {"split_by_div": True, "merge_coefficients": True}, [nd.Number(2.0), x]),
    ]
)
def test_split_by_mul(node, flags, expected):
    items = node.split_by_mul(**flags)
    assert set(map(str, items)) == set(map(str, expected))


def test_split_by_mul_inv_node_type():
    """
    专门验证 (1 / y) 分割后产生的是真正的 Inv 节点类型，
    而不是由于重载魔法方法意外生成的 Div(1, y)
    """
    expr = nd.Number(1.0) / y
    items = expr.split_by_mul(split_by_div=True)
    
    # 获取切分后的节点类型
    node_types = [type(item).__name__ for item in items]
    
    # 期望切分结果中包含 "Inv" 类型的节点，且不应存在 "Div"
    assert "Inv" in node_types, f"Expected Inv node, but got types: {node_types}"
    assert "Div" not in node_types, f"Did not expect Div node, but got types: {node_types}"