import pytest
import numpy as np
import nd2py as nd
from nd2py.search.ndformer import NDFormerConfig, NDFormerTokenizer
import warnings

np.random.seed(42)

x, y, z = nd.variables('x y z', nettype='scalar')
n = nd.Variable('n', nettype='node')
e = nd.Variable('e', nettype='edge')
s = nd.Variable('s', nettype='scalar')

@pytest.mark.parametrize("node,expected_tokens,expected_parents,expected_nettypes", [
    # 1. 基础单变量测试
    (s, ['SCALARVAR_4'], ['INDEX-ROOT'], ['NETTYPE-SCALAR']),
    
    # 2. 基础一元操作符
    (nd.sin(x), ['Sin', 'SCALARVAR_1'], ['INDEX-ROOT', 'INDEX-0'], ['NETTYPE-SCALAR', 'NETTYPE-SCALAR']),
    
    # 3. 跨网络类型操作 (aggr: scalar -> node)
    (nd.aggr(x), ['Aggr', 'SCALARVAR_1'], ['INDEX-ROOT', 'INDEX-0'], ['NETTYPE-NODE', 'NETTYPE-SCALAR']),
    
    # 4. 基础二元操作符
    (x + y, ['Add', 'SCALARVAR_1', 'SCALARVAR_2'], ['INDEX-ROOT', 'INDEX-0', 'INDEX-0'], ['NETTYPE-SCALAR', 'NETTYPE-SCALAR', 'NETTYPE-SCALAR']),
    
    # 5. 嵌套表达式 (深度 > 1) -> 测试 parent 索引是否能正确指向内部节点 (INDEX-1)
    # (x + y) * z 的前序遍历应该是: Mul, Add, x, y, z
    ((x + y) * z, 
     ['Mul', 'Add', 'SCALARVAR_1', 'SCALARVAR_2', 'SCALARVAR_3'], 
     ['INDEX-ROOT', 'INDEX-0', 'INDEX-1', 'INDEX-1', 'INDEX-0'], 
     ['NETTYPE-SCALAR', 'NETTYPE-SCALAR', 'NETTYPE-SCALAR', 'NETTYPE-SCALAR', 'NETTYPE-SCALAR']),

    # 6. 一元与二元组合嵌套
    # sin(x + y) 的前序遍历: Sin, Add, x, y
    (nd.sin(x + y),
     ['Sin', 'Add', 'SCALARVAR_1', 'SCALARVAR_2'],
     ['INDEX-ROOT', 'INDEX-0', 'INDEX-1', 'INDEX-1'],
     ['NETTYPE-SCALAR', 'NETTYPE-SCALAR', 'NETTYPE-SCALAR', 'NETTYPE-SCALAR']),

    # 7. 包含节点/边网络类型的复杂映射 (测试 Sour/Targ 等图神经网络特有操作)
    (nd.sour(n), 
     ['Sour', 'NODEVAR_1'], 
     ['INDEX-ROOT', 'INDEX-0'], 
     ['NETTYPE-EDGE', 'NETTYPE-NODE']),

    # 8. 【重要】常数节点 (Number) 长度对齐测试
    (x + nd.Number(1.0, nettype='scalar'),
     ['Add', 'SCALARVAR_1', '+', 'N1000', 'E+00'], 
     ['INDEX-ROOT', 'INDEX-0', 'INDEX-0'], 
     ['NETTYPE-SCALAR', 'NETTYPE-SCALAR', 'NETTYPE-SCALAR']),
])
def test_ndformer_tokenizer(node, expected_tokens, expected_parents, expected_nettypes):
    config = NDFormerConfig()
    tokenizer = NDFormerTokenizer(config, variables=[x, y, z, n, e, s])
    tokens, parents, nettypes = tokenizer.encode(node, mode='token')

    assert tokens == expected_tokens
    assert parents == expected_parents
    assert nettypes == expected_nettypes
    assert str(node) == str(tokenizer.decode(tokens, parents, nettypes))
