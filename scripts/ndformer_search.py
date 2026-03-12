# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
Debug script for NDFormer-guided MCTS symbolic regression

Tests the NDFormerMCTS search with a simple equation: x + sin(y)
Uses a randomly initialized NDFormer model (not trained)
"""
import logging
import torch
import numpy as np
import nd2py as nd
from nd2py.search.ndformer import NDFormerMCTS, NDFormerConfig, NDFormerModel, NDFormerTokenizer

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nd2py.ndformer_search')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
NUM_SAMPLES = 100
NUM_NODES = 20  # For graph-based problem
N_ITER = 50     # MCTS iterations
USE_CUDA = torch is not None and torch.cuda.is_available()

def main():
    logger.info("=" * 60)
    logger.info("NDFormer-guided MCTS Symbolic Regression Demo")
    logger.info("=" * 60)

    # ==========================================
    # 1. Create target equation: x + aggr(y)
    # ==========================================
    logger.info("\n[Step 1] Creating target equation: x + aggr(y)")

    x = nd.Variable('x', nettype='node')
    y = nd.Variable('y', nettype='edge')
    target_eq = nd.Add(x, nd.aggr(y))

    logger.info(f"Target equation: {target_eq}")
    logger.info(f"Target equation string: {target_eq.to_str()}")

    # ==========================================
    # 2. Generate synthetic data
    # ==========================================
    logger.info("\n[Step 2] Generating synthetic data")

    # Generate random graph-structured data
    # Create edge list for a simple graph
    edge_index = []
    for i in range(NUM_NODES):
        for j in range(i+1, NUM_NODES):
            edge_index.append((i, j))

    num_edges = len(edge_index)

    # Generate random input values
    X = {
        'x': np.random.uniform(-5, 5, (NUM_SAMPLES, NUM_NODES)),  # node features
        'y': np.random.uniform(-5, 5, (NUM_SAMPLES, num_edges)),  # edge features
    }

    # Compute target values
    edge_list = ([e[0] for e in edge_index], [e[1] for e in edge_index])
    y_true = target_eq.eval(X, edge_list=edge_list, num_nodes=NUM_NODES)

    logger.info(f"Data shape: X[x]={X['x'].shape}, X[y]={X['y'].shape}, y={y_true.shape}")
    logger.info(f"Target y range: [{y_true.min():.4f}, {y_true.max():.4f}]")

    # ==========================================
    # 3. Initialize NDFormer model (random weights)
    # ==========================================
    logger.info("\n[Step 3] Initializing NDFormer model (random weights, not trained)")

    config = NDFormerConfig()
    device = 'cuda' if USE_CUDA else 'cpu'
    logger.info(f"Using device: {device}")

    tokenizer = NDFormerTokenizer(config, variables=[x, y])
    model = NDFormerModel(config, tokenizer).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==========================================
    # 4. Initialize NDFormerMCTS
    # ==========================================
    logger.info("\n[Step 4] Initializing NDFormerMCTS")

    # Build edge list for MCTS
    edge_list = ([e[0] for e in edge_index], [e[1] for e in edge_index])

    search = NDFormerMCTS(
        variables=[x, y],
        binary=[nd.Add],  # Only Add
        unary=[nd.Aggr],   # Only Aggr
        max_params=0,      # No constants needed
        const_range=(-2.0, 2.0),
        depth_range=(2, 4),  # Reduced depth since equation is simple: x + aggr(y) has depth ~3
        nettype='node',
        num_nodes=NUM_NODES,
        edge_list=edge_list,
        n_iter=N_ITER,
        use_tqdm=True,
        child_num=20,
        n_playout=50,
        d_playout=5,
        max_len=6,  # Short sequence: [x, y, aggr, Add] ~ 4-6 tokens
        c=1.41,
        eta=0.99,
        # NDFormer parameters
        ndformer=model,
        ndformer_tokenizer=tokenizer,
        puct_c_puct=1.0,
        ndformer_topk=8,
        ndformer_temperature=1.0,
    )

    # ==========================================
    # 5. Run MCTS search
    # ==========================================
    logger.info("\n[Step 5] Running NDFormer-guided MCTS search")
    logger.info("-" * 40)

    search.fit(X, y_true)

    logger.info("-" * 40)
    logger.info("\n[Search Finished]")

    # ==========================================
    # 6. Report results
    # ==========================================
    logger.info("\n[Step 6] Results")
    logger.info("=" * 60)

    result_eq = search.eqtree
    logger.info(f"Discovered equation: {result_eq}")
    logger.info(f"Discovered equation string: {result_eq.to_str()}")

    # Evaluate discovered equation
    y_pred = result_eq.eval(X, edge_list=edge_list, num_nodes=NUM_NODES)

    # Compute metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_pred - y_true) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10)

    logger.info(f"\nMetrics:")
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²:   {r2:.6f}")

    # Check if exact match
    if str(result_eq) == str(target_eq):
        logger.info("\n[SUCCESS] Exact match found!")
    elif r2 > 0.99:
        logger.info("\n[GOOD] High accuracy match found!")
    elif r2 > 0.9:
        logger.info("\n[OK] Reasonable approximation found!")
    else:
        logger.info("\n[INFO] Search did not find a good solution.")
        logger.info("This is expected with a randomly initialized NDFormer.")

    # Show search history
    if search.records:
        logger.info(f"\nSearch history ({len(search.records)} iterations):")
        best_record = max(search.records, key=lambda r: r.get('reward', -float('inf')))
        logger.info(f"Best iteration: {best_record.get('iter', 'N/A')}")
        logger.info(f"Best equation: {best_record.get('fitted_eqtree', 'N/A')}")
        logger.info(f"Best R²: {best_record.get('r2', 'N/A'):.6f}")
        logger.info(f"Best reward: {best_record.get('reward', 'N/A'):.6f}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo finished!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
