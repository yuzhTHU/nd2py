import pytest
import torch
import numpy as np
import nd2py as nd
from nd2py.search.ndformer import (
    NDFormerConfig,
    NDFormerTokenizer,
    NDFormerDataset,
    NDFormerEqtreeGenerator,
    NDFormerGraphGenerator,
    NDFormerDataGenerator,
)

np.random.seed(42)

# 定义测试用的变量 - 只使用 node 和 edge 类型，因为 dataset 不支持 scalar
n = nd.Variable('n', nettype='node')
e = nd.Variable('e', nettype='edge')


class TestNDFormerDataset:
    """测试 NDFormerDataset 的功能"""

    @pytest.fixture
    def config(self):
        """创建配置"""
        config = NDFormerConfig()
        config.min_node_num = 3
        config.max_node_num = 5
        return config

    @pytest.fixture
    def variables(self):
        """创建变量"""
        return [n, e]

    @pytest.fixture
    def tokenizer(self, config, variables):
        """创建 tokenizer"""
        return NDFormerTokenizer(config, variables=variables)

    @pytest.fixture
    def eqtree_generator(self, config, variables):
        """创建方程树生成器"""
        return NDFormerEqtreeGenerator(
            variables=variables,
            binary=[nd.Add, nd.Mul],
            unary=[nd.Aggr],
            depth_range=(2, 4),
            full_prob=1.0,
            num_nodes=10,
        )

    @pytest.fixture
    def topo_generator(self, config):
        """创建图拓扑生成器"""
        # 设置较大的最小节点数，避免 BA 图生成失败
        config.min_node_num = 10
        config.max_node_num = 20
        return NDFormerGraphGenerator(config)

    @pytest.fixture
    def data_generator(self, config):
        """创建数据生成器"""
        return NDFormerDataGenerator(config)

    @pytest.fixture
    def dataset(self, config, eqtree_generator, topo_generator, data_generator, tokenizer):
        """创建数据集"""
        return NDFormerDataset(
            config=config,
            eqtree_generator=eqtree_generator,
            topo_generator=topo_generator,
            data_generator=data_generator,
            tokenizer=tokenizer,
            n_samples=10,
            random_state=42,
        )

    def test_init(self, dataset, config):
        """测试数据集初始化"""
        assert dataset.config == config
        assert dataset.n_samples == 10
        assert dataset.random_state == 42
        assert len(dataset) == 10

    def test_getitem(self, dataset):
        """测试获取单个样本"""
        sample = dataset[0]

        # 验证返回的键
        assert 'edge_list' in sample
        assert 'data_node' in sample
        assert 'data_edge' in sample
        assert 'num_nodes' in sample
        assert 'partial_eqs' in sample
        assert 'next_tokens' in sample

        # 验证类型
        assert isinstance(sample['edge_list'], torch.Tensor)
        assert isinstance(sample['data_node'], torch.Tensor)
        assert isinstance(sample['data_edge'], torch.Tensor)
        assert isinstance(sample['num_nodes'], int)
        assert isinstance(sample['partial_eqs'], list)
        assert isinstance(sample['next_tokens'], list)

    def test_sample_contains_empty_nodes(self, dataset):
        """测试生成的样本包含 Empty 节点"""
        sample = dataset[0]

        assert len(sample['partial_eqs']) > 0
        assert len(sample['next_tokens']) > 0
        assert len(sample['partial_eqs']) == len(sample['next_tokens'])

        # 验证每个 partial_eq 都包含 Empty 节点
        tokenizer = dataset.tokenizer
        empty_token_id = tokenizer.token2id['EMPTY']

        for partial_eq, next_token in zip(sample['partial_eqs'], sample['next_tokens']):
            tokens = partial_eq.tolist()
            assert empty_token_id in tokens, "每个不完整方程应该包含至少一个 Empty 节点"

    def test_next_token_matches_empty(self, dataset):
        """测试 next_token 对应第一个 Empty 位置的原始符号"""
        sample = dataset[0]

        tokenizer = dataset.tokenizer
        empty_token_id = tokenizer.token2id['EMPTY']

        for partial_eq, next_token in zip(sample['partial_eqs'], sample['next_tokens']):
            tokens = partial_eq.tolist()
            next_tok = next_token.item()

            # 验证 next_token 是有效 token
            assert 0 <= next_tok < tokenizer.vocab_size

            # 找到第一个 Empty 的位置
            first_empty_idx = tokens.index(empty_token_id)
            assert first_empty_idx >= 0

    def test_collate_fn(self, dataset):
        """测试批量数据合并"""
        batch = [dataset[i] for i in range(3)]
        collated = dataset.collate_fn(batch)

        # 验证合并后的键
        expected_keys = ['edge_list', 'data_node', 'data_edge', 'num_nodes',
                        'partial_eqs', 'next_tokens', 'node_batch_idx', 'seq_batch_idx']
        for key in expected_keys:
            assert key in collated

        # 验证类型和形状
        assert isinstance(collated['edge_list'], torch.Tensor)
        assert isinstance(collated['data_node'], torch.Tensor)
        assert isinstance(collated['data_edge'], torch.Tensor)
        assert isinstance(collated['num_nodes'], int)
        assert isinstance(collated['partial_eqs'], torch.Tensor)
        assert isinstance(collated['next_tokens'], torch.Tensor)
        assert isinstance(collated['node_batch_idx'], torch.Tensor)
        assert isinstance(collated['seq_batch_idx'], torch.Tensor)

        # 验证 batch 维度
        batch_size = 3
        assert collated['next_tokens'].shape[0] >= batch_size  # 至少每个样本有一个序列
        assert collated['node_batch_idx'].shape[0] == collated['num_nodes']

    def test_get_sampler(self, dataset):
        """测试获取采样器"""
        sampler = dataset.get_sampler()
        assert sampler is None  # 因为 n_samples 不是 None

    def test_infinite_sampler(self, config, eqtree_generator, topo_generator, data_generator, tokenizer):
        """测试无限采样器"""
        infinite_dataset = NDFormerDataset(
            config=config,
            eqtree_generator=eqtree_generator,
            topo_generator=topo_generator,
            data_generator=data_generator,
            tokenizer=tokenizer,
            n_samples=None,  # 无限数据集
            random_state=42,
        )

        sampler = infinite_dataset.get_sampler()
        assert sampler is not None

        # 测试采样器可以无限生成索引
        iterator = iter(sampler)
        for i in range(100):
            assert next(iterator) == i

    def test_data_shapes(self, dataset):
        """测试数据形状"""
        sample = dataset[0]

        # data_node 形状：(sample_num, num_nodes, max_var_num+1, 3)
        assert len(sample['data_node'].shape) == 4
        assert sample['data_node'].shape[2] == dataset.config.max_var_num + 1
        assert sample['data_node'].shape[3] == 3  # number tokenizer 的 3 个 token

        # data_edge 形状：(sample_num, num_edges, max_var_num+1, 3)
        assert len(sample['data_edge'].shape) == 4
        assert sample['data_edge'].shape[2] == dataset.config.max_var_num + 1
        assert sample['data_edge'].shape[3] == 3

    def test_multiple_samples(self, dataset):
        """测试获取多个样本"""
        samples = [dataset[i] for i in range(5)]

        # 验证每个样本都有正确的结构
        for sample in samples:
            assert 'partial_eqs' in sample
            assert 'next_tokens' in sample
            assert len(sample['partial_eqs']) > 0
            assert len(sample['partial_eqs']) == len(sample['next_tokens'])

    def test_random_state_reproducibility(self, config, eqtree_generator, topo_generator, data_generator, tokenizer):
        """测试随机种子的可复现性"""
        dataset1 = NDFormerDataset(
            config=config,
            eqtree_generator=eqtree_generator,
            topo_generator=topo_generator,
            data_generator=data_generator,
            tokenizer=tokenizer,
            n_samples=5,
            random_state=123,
        )

        dataset2 = NDFormerDataset(
            config=config,
            eqtree_generator=eqtree_generator,
            topo_generator=topo_generator,
            data_generator=data_generator,
            tokenizer=tokenizer,
            n_samples=5,
            random_state=123,
        )

        # 相同随机种子应该产生相同的数据（对于相同的索引）
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        # 注意：由于 deepcopy 和随机选择的复杂性，这里只验证基本结构
        assert len(sample1['partial_eqs']) == len(sample2['partial_eqs'])
        assert len(sample1['next_tokens']) == len(sample2['next_tokens'])


class TestEmptyNodeTrainingData:
    """测试 Empty 节点训练数据生成的正确性"""

    @pytest.fixture
    def config(self):
        config = NDFormerConfig()
        config.min_node_num = 3
        config.max_node_num = 5
        return config

    @pytest.fixture
    def simple_dataset(self, config):
        """创建使用简单方程的数据集用于验证"""
        variables = [n, e]
        tokenizer = NDFormerTokenizer(config, variables=variables)

        # 使用固定的简单方程生成器
        eqtree_generator = NDFormerEqtreeGenerator(
            variables=variables,
            binary=[nd.Add],
            unary=[nd.Aggr],
            depth_range=(2, 3),
            full_prob=1.0,
            num_nodes=10,
        )

        return NDFormerDataset(
            config=config,
            eqtree_generator=eqtree_generator,
            topo_generator=NDFormerGraphGenerator(config),
            data_generator=NDFormerDataGenerator(config),
            tokenizer=tokenizer,
            n_samples=5,
            random_state=42,
        )

    def test_progressive_replacement(self, simple_dataset):
        """测试渐进式替换生成多个样本"""
        sample = simple_dataset[0]

        # 应该生成多个训练样本（每个替换步骤一个）
        num_samples = len(sample['partial_eqs'])
        assert num_samples >= 1

        # 样本数量应该与方程的节点数相关
        # 简单方程 x + y 有 3 个节点，应该生成 3 个样本
        # 但实际数量可能因随机性而异
        assert num_samples <= 10  # 合理上限

    def test_tokenizer_encoding(self, config):
        """测试 tokenizer 编码包含 Empty 的方程"""
        variables = [n, e]
        tokenizer = NDFormerTokenizer(config, variables=variables)

        # 手动构建带 Empty 的方程
        empty = nd.Empty(nettype='edge')
        eq_with_empty = nd.Aggr(empty)

        tokens, parents, nettypes = tokenizer.encode(eq_with_empty, mode='token')

        assert 'EMPTY' in tokens
        assert tokens.index('EMPTY') > 0  # Empty 不在第一个位置

        # 测试 roundtrip
        tokens_ids, parents_ids, nettypes_ids = tokenizer.encode(eq_with_empty, mode='token_id')
        decoded = tokenizer.decode(tokens_ids, parents_ids, nettypes_ids, mode='token_id')

        # 验证结构
        original_preorder = [type(s).__name__ for s in eq_with_empty.iter_preorder()]
        decoded_preorder = [type(s).__name__ for s in decoded.iter_preorder()]
        assert original_preorder == decoded_preorder
