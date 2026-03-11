# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
Debug script for NDFormer-guided MCTS symbolic regression

Tests the NDFormerMCTS search with a simple equation: x + sin(y)
Uses a randomly initialized NDFormer model (not trained)
"""
import logging
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
torch = None
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
except ImportError:
    pass

# Configuration
NUM_SAMPLES = 100
NUM_NODES = 20  # For graph-based problem
N_ITER = 50     # MCTS iterations
USE_CUDA = torch and torch.cuda.is_available()

def main():
    logger.info("=" * 60)
    logger.info("NDFormer-guided MCTS Symbolic Regression Demo")
    logger.info("=" * 60)

    # ==========================================
    # 1. Create target equation: x + sin(y)
    # ==========================================
    logger.info("\n[Step 1] Creating target equation: x + sin(y)")

    x = nd.Variable('x', nettype='scalar')
    y = nd.Variable('y', nettype='scalar')
    target_eq = nd.Add(x, nd.Sin(y))

    logger.info(f"Target equation: {target_eq}")
    logger.info(f"Target equation string: {target_eq.to_str()}")

    # ==========================================
    # 2. Generate synthetic data
    # ==========================================
    logger.info("\n[Step 2] Generating synthetic data")

    # Generate random input values
    X = {
        'x': np.random.uniform(-5, 5, NUM_SAMPLES),
        'y': np.random.uniform(-5, 5, NUM_SAMPLES),
    }

    # Compute target values
    y_true = target_eq.eval(X)

    logger.info(f"Data shape: X[x]={X['x'].shape}, X[y]={X['y'].shape}, y={y_true.shape}")
    logger.info(f"Target y range: [{y_true.min():.4f}, {y_true.max():.4f}]")

    # ==========================================
    # 3. Initialize NDFormer model (random weights)
    # ==========================================
    logger.info("\n[Step 3] Initializing NDFormer model (random weights, not trained)")

    config = NDformerConfig()
    device = 'cuda' if USE_CUDA else 'cpu'
    logger.info(f"Using device: {device}")

    model = NDformerModel(config).to(device)
    tokenizer = NDFormerTokenizer(config, variables=[x, y])

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ==========================================
    # 4. Initialize NDFormerMCTS
    # ==========================================
    logger.info("\n[Step 4] Initializing NDFormerMCTS")

    search = NDFormerMCTS(
        variables=[x, y],
        binary=[nd.Add, nd.Sub, nd.Mul, nd.Div],
        unary=[nd.Sin, nd.Cos, nd.Abs, nd.Neg, nd.Sqrt],
        max_params=2,
        const_range=(-2.0, 2.0),
        depth_range=(2, 8),
        nettype='scalar',
        n_iter=N_ITER,
        use_tqdm=True,
        child_num=20,
        n_playout=50,
        d_playout=5,
        max_len=20,
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
    y_pred = result_eq.eval(X)

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
