# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
Debug script for NDFormer-guided MCTS symbolic regression

Tests the NDFormerMCTS search with a simple equation: x + aggr(y)
Uses a randomly initialized NDFormer model (not trained)
"""
import logging
import sys
import json
import random
import re
import torch
import numpy as np
import nd2py as nd
import shlex
from datetime import datetime
from pathlib import Path
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from nd2py.search.ndformer import NDFormerMCTS, NDFormerConfig, NDFormerModel, NDFormerTokenizer

_logger = logging.getLogger("nd2py.ndformer_search")



def generate_data(variables: list[nd.Variable], num_nodes: int, num_edges: int, n_samples: int) -> dict:
    """Generate synthetic data for given variables."""
    X = {}
    for var in variables:
        if var.nettype == 'node':
            X[var.name] = np.random.uniform(-5, 5, (n_samples, num_nodes))
        elif var.nettype == 'edge':
            X[var.name] = np.random.uniform(-5, 5, (n_samples, num_edges))
        elif var.nettype == 'scalar':
            X[var.name] = np.random.uniform(-5, 5, (n_samples,))
        else:
            raise ValueError(f"Unknown nettype: {var.nettype}")
    return X


def main(args):
    # Parse operator strings to actual operator objects
    op_map = {
        'Add': nd.Add, 'Sub': nd.Sub, 'Mul': nd.Mul, 'Div': nd.Div,
        'Aggr': nd.Aggr, 'Rgga': nd.Rgga, 'Sour': nd.Sour, 'Targ': nd.Targ,
        'Sin': nd.Sin, 'Cos': nd.Cos, 'Tan': nd.Tan,
        'Abs': nd.Abs, 'Neg': nd.Neg, 'Sqrt': nd.Sqrt, 'Log': nd.Log,
    }

    # Default operators
    if args.binary_ops is None:
        args.binary_ops = [nd.Add]
    else:
        args.binary_ops = [op_map[op] for op in args.binary_ops]

    if args.unary_ops is None:
        args.unary_ops = [nd.Aggr]
    else:
        args.unary_ops = [op_map[op] for op in args.unary_ops]

    # 1. Define variables with correct nettypes
    # For equation "x + aggr(y)": x is node, y is edge
    variables = [
        nd.Variable('x', nettype='node'),
        nd.Variable('y', nettype='edge'),
    ]
    var_dict = {var.name: var for var in variables}

    # 2. Create graph topology
    num_nodes = args.num_nodes
    edge_list = ([], [])
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edge_list[0].append(i)
            edge_list[1].append(j)
    num_edges = len(edge_list[0])

    # 3. Build target equation with correct variables
    target_eq = nd.parse(args.eq, variables=var_dict)

    # 4. Generate synthetic data
    X = generate_data(variables, num_nodes, num_edges, args.n_samples)
    y_true = target_eq.eval(X, edge_list=edge_list, num_nodes=num_nodes)

    _logger.info(
        f"Target equation: {target_eq.to_str()}\n"
        f"Data shapes: {[(k, v.shape) for k, v in X.items()]}, y={y_true.shape}\n"
        f"Target y range: [{y_true.min():.4f}, {y_true.max():.4f}]"
    )

    # 5. Initialize NDFormer model and load trained weights
    config = NDFormerConfig()
    device = args.device
    tokenizer = NDFormerTokenizer(config, variables=variables)
    model = NDFormerModel(config, tokenizer).to(device)

    # Load trained checkpoint if provided
    if args.ndformer_ckpt is not None:
        ckpt_path = Path(args.ndformer_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint['model']
        # Handle both 'model' dict and 'model_state_dict' key formats
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            model_state = model_state['model_state_dict']
        model.load_state_dict(model_state)
        _logger.note(f"Loaded trained NDFormer from {ckpt_path}")

    model.eval()  # Set to evaluation mode
    _logger.info(f"Using device: {device}")
    _logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Initialize NDFormerMCTS
    search = NDFormerMCTS(
        variables=variables,
        binary=args.binary_ops,
        unary=args.unary_ops,
        max_params=args.max_params,
        const_range=(-2.0, 2.0),
        depth_range=(2, 6),
        nettype='node',
        num_nodes=num_nodes,
        edge_list=edge_list,
        n_iter=args.n_iter,
        use_tqdm=True,
        child_num=20,
        n_playout=50,
        d_playout=5,
        max_len=10,
        c=1.41,
        eta=0.99,
        # NDFormer parameters
        ndformer=model,
        ndformer_tokenizer=tokenizer,
        puct_c_puct=1.0,
        ndformer_topk=8,
        ndformer_temperature=1.0,
        beam_width=args.beam_width,
    )
    _logger.info(f"NDFormerMCTS initialized with operators: binary={args.binary_ops}, unary={args.unary_ops}")

    # 7. Run MCTS search
    _logger.info("\n" + "=" * 60)
    _logger.info("Starting NDFormer-guided MCTS search")
    _logger.info("=" * 60 + "\n")

    search.fit(X, y_true)

    # 8. Report results
    _logger.info("\n" + "=" * 60)
    _logger.info("Results")
    _logger.info("=" * 60)

    result_eq = search.eqtree
    _logger.info(f"Discovered equation: {result_eq}")
    _logger.info(f"Discovered equation string: {result_eq.to_str()}")

    # Evaluate discovered equation
    y_pred = result_eq.eval(X, edge_list=edge_list, num_nodes=num_nodes)

    # Compute metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_pred - y_true) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)

    _logger.info(f"\nMetrics:")
    _logger.info(f"  MSE:  {mse:.6f}")
    _logger.info(f"  RMSE: {rmse:.6f}")
    _logger.info(f"  R²:   {r2:.6f}")

    # Check if exact match
    if str(result_eq) == str(target_eq):
        _logger.info("\n[SUCCESS] Exact match found!")
    elif r2 > 0.99:
        _logger.info("\n[GOOD] High accuracy match found!")
    elif r2 > 0.9:
        _logger.info("\n[OK] Reasonable approximation found!")
    else:
        _logger.info("\n[INFO] Search did not find a good solution.")
        _logger.info("This is expected with a randomly initialized NDFormer.")

    # Show search history
    if search.records:
        _logger.info(f"\nSearch history ({len(search.records)} iterations):")
        best_record = max(search.records, key=lambda r: r.get('reward', -float('inf')))
        _logger.info(f"Best iteration: {best_record.get('iter', 'N/A')}")
        _logger.info(f"Best equation: {best_record.get('fitted_eqtree', 'N/A')}")
        _logger.info(f"Best R²: {best_record.get('r2', 'N/A'):.6f}")
        _logger.info(f"Best reward: {best_record.get('reward', 'N/A'):.6f}")

    _logger.info("\n" + "=" * 60)
    _logger.info("Demo finished!")
    _logger.info("=" * 60)


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="search", help="实验任务名称，用于生成实验 ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--random_state", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/search", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    parser.add_argument("--minimize_gpu", action="store_true", default=False, help="是否在每个 epoch 结束后尽可能释放显存以供其他进程使用")

    # Equation and data configuration
    parser.add_argument("--eq", type=str, default="x + aggr(y)", help="Target equation to discover")
    parser.add_argument("--num_nodes", type=int, default=20, help="Number of nodes in the graph")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of data samples")

    # Search operators
    parser.add_argument("--binary_ops", type=str, nargs="+", default=None, help="Binary operators")
    parser.add_argument("--unary_ops", type=str, nargs="+", default=None, help="Unary operators")
    parser.add_argument("--max_params", type=int, default=0, help="Maximum number of numeric constants")

    # MCTS configuration
    parser.add_argument("--n_iter", type=int, default=50, help="Number of MCTS iterations")
    parser.add_argument('--beam_width', type=int, default=10, help="Beam width for batch expansion")
    parser.add_argument('--ndformer_ckpt', type=str, default=None, help="Path to trained NDFormer checkpoint (e.g., logs/train/xxx/best.pth)")
    parser.add_argument('--required_memory_MB', type=int, default=5000, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

    parser = nd.utils.add_minus_flags(parser) ## --key_name -> --key-name
    parser = nd.utils.add_negation_flags(parser) ## --action-as-true -> --no-action-as-true
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        curr = now.strftime("%H%M%S")
        host = gethostname()
        exp_name = f'{date}_{args.name}_{curr}_{host}'
        exp_name = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub('_', exp_name.strip())
        exp_name = exp_name or 'unnamed'
        exp_name = exp_name[:255] # Max filename length on most filesystems
        args.exp_name = exp_name
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Init Logger
    nd.utils.init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )
    nd.utils.init_logger(
        "nd2py",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    ## Warn Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Set random_state
    if args.random_state is None:
        args.random_state = random.randint(1, 10000)
    nd.utils.seed_all(args.random_state)
    ## Set Command
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))
    ## Select GPU
    if args.device == "auto":
        args.device = nd.utils.AutoGPU().choice_gpu(memory_MB=args.required_memory_MB, interval=15)

    ## Save Args
    args_path = save_path / "args.json"
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists(): i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    _logger.note(f"Args: {args}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    ## Start Search
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
