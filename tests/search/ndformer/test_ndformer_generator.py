import pytest
import numpy as np
import nd2py as nd
from nd2py.search.ndformer import (
    NDFormerConfig,
    NDFormerEqtreeGenerator,
    NDFormerGraphGenerator,
    NDFormerDataGenerator,
)

np.random.seed(42)

# 定义测试用的变量
x, y, z = nd.variables('x y z', nettype='scalar')
n = nd.Variable('n', nettype='node')
e = nd.Variable('e', nettype='edge')
s = nd.Variable('s', nettype='scalar')


class TestNDFormerEqtreeGenerator:
    """测试 NDFormerEqtreeGenerator 的功能"""

    def test_init(self):
        """测试初始化参数"""
        config = NDFormerConfig()
        variables = [x, y, z]
        generator = NDFormerEqtreeGenerator(variables=variables)

        assert generator.variables == variables
        assert generator.depth_range == (2, 6)  # 默认值

    def test_sample_scalar_nettype(self):
        """测试标量类型的方程生成"""
        variables = [x, y, z]
        generator = NDFormerEqtreeGenerator(
            variables=variables,
            binary=[nd.Add, nd.Mul],
            unary=[nd.Sin],
            depth_range=(2, 4),
            full_prob=1.0,  # 强制生成满树
        )
        rng = np.random.default_rng(42)

        eqtree = generator.sample(nettypes={'scalar'}, _rng=rng)

        # 验证生成的树有效
        assert eqtree is not None
        assert eqtree.nettype == 'scalar'
        # 验证树深度在范围内
        depth = self._get_tree_depth(eqtree)
        assert 2 <= depth <= 4

    def test_sample_node_nettype(self):
        """测试节点类型的方程生成"""
        variables = [n]
        generator = NDFormerEqtreeGenerator(
            variables=variables,
            binary=[nd.Add],
            unary=[nd.Aggr],
            depth_range=(2, 4),
            num_nodes=10,
        )
        rng = np.random.default_rng(42)

        eqtree = generator.sample(nettypes={'node'}, _rng=rng)

        assert eqtree is not None
        assert eqtree.nettype == 'node'

    def test_sample_mixed_nettypes(self):
        """测试混合网络类型的方程生成"""
        variables = [n, e, s]
        generator = NDFormerEqtreeGenerator(
            variables=variables,
            binary=[nd.Add],
            unary=[nd.Aggr, nd.Sour, nd.Targ],
            depth_range=(2, 5),
            num_nodes=10,
        )
        rng = np.random.default_rng(42)

        eqtree = generator.sample(nettypes={'node', 'edge', 'scalar'}, _rng=rng)

        assert eqtree is not None
        assert eqtree.nettype in {'node', 'edge', 'scalar'}

    def test_sample_with_constants(self):
        """测试包含数值常数的方程生成"""
        variables = [x]
        generator = NDFormerEqtreeGenerator(
            variables=variables,
            binary=[nd.Add, nd.Mul],
            unary=[],
            depth_range=(2, 3),
            const_range=(-1.0, 1.0),
            full_prob=1.0,
        )
        rng = np.random.default_rng(42)

        eqtree = generator.sample(nettypes={'scalar'}, _rng=rng)

        # 验证树中包含 Number 节点
        has_number = any(isinstance(node, nd.Number) for node in eqtree.iter_preorder())
        assert has_number

    def _get_tree_depth(self, node):
        """辅助函数：计算树的深度"""
        if node.n_operands == 0:
            return 1
        return 1 + max(self._get_tree_depth(op) for op in node.operands)


class TestNDFormerGraphGenerator:
    """测试 NDFormerGraphGenerator 的功能"""

    def test_init(self):
        """测试初始化参数"""
        config = NDFormerConfig()
        generator = NDFormerGraphGenerator(config)

        assert generator.min_node_num == config.min_node_num
        assert generator.max_node_num == config.max_node_num

    @pytest.mark.parametrize("topology", ['ER', 'BA', 'Complete'])
    def test_sample_topologies(self, topology):
        """测试不同图拓扑结构的生成"""
        config = NDFormerConfig()
        config.min_node_num = 5
        config.max_node_num = 10
        generator = NDFormerGraphGenerator(config)
        rng = np.random.default_rng(42)

        edge_list, num_nodes = generator.sample(topology=topology, _rng=rng)

        # 验证返回格式 (edge_list 是 list of lists, not tuple)
        assert isinstance(edge_list, list)
        assert len(edge_list) == 2
        assert isinstance(num_nodes, int)
        assert num_nodes >= 5
        assert num_nodes <= 10

        # 验证边列表格式
        assert len(edge_list[0]) == len(edge_list[1])
        assert all(0 <= src < num_nodes for src in edge_list[0])
        assert all(0 <= dst < num_nodes for dst in edge_list[1])

    def test_er_graph_parameters(self):
        """测试 ER 图参数"""
        config = NDFormerConfig()
        generator = NDFormerGraphGenerator(config)
        rng = np.random.default_rng(42)

        # 测试指定节点数
        edge_list, num_nodes = generator.generate_ER_graph(V=8, _rng=rng)
        assert num_nodes == 8

        # 测试指定边数范围
        edge_list, num_nodes = generator.generate_ER_graph(V=10, E=20, _rng=rng)
        assert len(edge_list[0]) >= 15  # 允许一定的随机性

    def test_ba_graph_parameters(self):
        """测试 BA 图参数"""
        config = NDFormerConfig()
        generator = NDFormerGraphGenerator(config)
        rng = np.random.default_rng(42)

        edge_list, num_nodes = generator.generate_BA_graph(V=10, m=2, _rng=rng)

        assert num_nodes == 10
        # BA 图的边数至少为 (V-1) * m
        assert len(edge_list[0]) >= 9

    def test_complete_graph(self):
        """测试完全图生成"""
        config = NDFormerConfig()
        generator = NDFormerGraphGenerator(config)
        rng = np.random.default_rng(42)

        edge_list, num_nodes = generator.generate_complete_graph(V=5, _rng=rng)

        assert num_nodes == 5
        # 完全图的边数 = n*(n-1)/2 (无向图)
        expected_edges = num_nodes * (num_nodes - 1) // 2
        assert len(edge_list[0]) == expected_edges


class TestNDFormerDataGenerator:
    """测试 NDFormerDataGenerator 的功能"""

    def test_init(self):
        """测试初始化参数"""
        config = NDFormerConfig()
        generator = NDFormerDataGenerator(config)

        assert generator.min_var_val == config.min_var_val
        assert generator.max_var_val == config.max_var_val

    def test_sample_uniform_data(self):
        """测试均匀分布数据生成"""
        config = NDFormerConfig()
        generator = NDFormerDataGenerator(config)
        rng = np.random.default_rng(42)

        eqtree = x + y
        edge_list, num_nodes = ([], []), 0

        var_dict, target = generator.sample(
            eqtree=eqtree,
            edge_list=edge_list,
            num_nodes=num_nodes,
            sample_num=10,
            dist_type='Uniform',
            _rng=rng,
        )

        assert len(var_dict['x']) == 10
        assert len(var_dict['y']) == 10
        assert len(target) == 10
        # 验证数据在合理范围内
        assert np.all(np.isfinite(var_dict['x']))
        assert np.all(np.isfinite(target))

    def test_sample_gaussian_data(self):
        """测试高斯分布数据生成"""
        config = NDFormerConfig()
        generator = NDFormerDataGenerator(config)
        rng = np.random.default_rng(42)

        eqtree = nd.sin(x)
        edge_list, num_nodes = ([], []), 0

        var_dict, target = generator.sample(
            eqtree=eqtree,
            edge_list=edge_list,
            num_nodes=num_nodes,
            sample_num=20,
            dist_type='Gaussian',
            _rng=rng,
        )

        assert len(var_dict['x']) == 20
        assert len(target) == 20
        assert np.all(np.isfinite(var_dict['x']))

    def test_sample_gmm_data(self):
        """测试 GMM 数据生成"""
        config = NDFormerConfig()
        generator = NDFormerDataGenerator(config)
        rng = np.random.default_rng(42)

        eqtree = x * y + z
        edge_list, num_nodes = ([], []), 0

        var_dict, target = generator.sample(
            eqtree=eqtree,
            edge_list=edge_list,
            num_nodes=num_nodes,
            sample_num=15,
            dist_type='GMM',
            _rng=rng,
        )

        assert len(var_dict['x']) == 15
        assert len(var_dict['y']) == 15
        assert len(var_dict['z']) == 15
        assert np.all(np.isfinite(target))

    def test_sample_with_node_variables(self):
        """测试包含节点变量的数据生成"""
        config = NDFormerConfig()
        config.min_node_num = 5
        config.max_node_num = 10
        generator = NDFormerDataGenerator(config)
        rng = np.random.default_rng(42)

        # 使用合法的节点类型方程：aggr(e) 是 node 类型
        eqtree = n + nd.Aggr(e)
        edge_list = ([0, 1, 2], [1, 2, 0])
        num_nodes = 3

        var_dict, target = generator.sample(
            eqtree=eqtree,
            edge_list=edge_list,
            num_nodes=num_nodes,
            sample_num=10,
            dist_type='Uniform',
            _rng=rng,
        )

        assert 'n' in var_dict
        assert 'e' in var_dict
        assert var_dict['n'].shape == (10, num_nodes)
        assert var_dict['e'].shape == (10, len(edge_list[0]))
        assert target.shape == (10, num_nodes)

    def test_invalid_dist_type(self):
        """测试无效分布类型的错误处理"""
        config = NDFormerConfig()
        generator = NDFormerDataGenerator(config)
        rng = np.random.default_rng(42)

        eqtree = x + y
        edge_list, num_nodes = ([], []), 0

        with pytest.raises(ValueError, match='Unknown data generation dist_type'):
            generator.sample(
                eqtree=eqtree,
                edge_list=edge_list,
                num_nodes=num_nodes,
                sample_num=10,
                dist_type='Invalid',
                _rng=rng,
            )
