# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
Debug script for NDFormer-guided MCTS symbolic regression

Tests the NDFormerMCTS search with a simple equation: x + aggr(y)
Uses a randomly initialized NDFormer model (not trained)
"""
import logging
import torch
import numpy as np
import nd2py as nd
from argparse import ArgumentParser
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

    # 5. Initialize NDFormer model (random weights)
    config = NDFormerConfig()
    device = args.device
    tokenizer = NDFormerTokenizer(config, variables=variables)
    model = NDFormerModel(config, tokenizer).to(device)
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

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

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

    main(args)
