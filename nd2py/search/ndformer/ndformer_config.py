# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass, field


@dataclass
class NDFormerModelConfig:
    """
    Configuration for NDFormer model architecture and capabilities.

    ═══════════════════════════════════════════════════════════════════════════
    PURPOSE
    ═══════════════════════════════════════════════════════════════════════════

    This class defines the **model's structure and capabilities**:

    - Model architecture (transformer layers, GNN layers, embedding dimensions)
    - Tokenization scheme (number encoding, vocabulary)
    - Supported operators and sequence length limits

    ═══════════════════════════════════════════════════════════════════════════
    RELATIONSHIP WITH NDFormerMCTS (INFERENCE-TIME SEARCH)
    ═══════════════════════════════════════════════════════════════════════════

    **TL;DR: Users of NDFormerMCTS do not need to interact with this class directly.**

    When using a pre-trained model with NDFormerMCTS:

    1. **The trained model + config is a black box**: The model and its
       associated config are loaded together from a checkpoint. The config
       is used internally to reconstruct the tokenizer and model architecture.

    2. **No control over search behavior**: NDFormerMCTS does NOT use this
       config to control how search proceeds. Search parameters (beam_width,
       temperature, c, etc.) are configured directly in NDFormerMCTS.__init__().

    3. **Capability validation only**: NDFormerMCTS may use the config to
       verify that search settings are within the model's capabilities:
       - max_len (search) should not exceed max_seq_len (model capability)
       - Operator set should be compatible with trained vocabulary
       - Variable count should not exceed max_var_num

    This design follows standard ML practice where model architecture config
    is separate from inference/search hyperparameters.

    ═══════════════════════════════════════════════════════════════════════════
    USAGE
    ═══════════════════════════════════════════════════════════════════════════

    Training a new model:
    ```python
    config = NDFormerModelConfig(model='default', n_head=16, d_emb=256)
    tokenizer = NDFormerTokenizer(config, variables)
    model = NDFormerModel.create(config, tokenizer)
    # ... train on dataset ...
    torch.save({'model': model.state_dict(), 'config': config}, 'checkpoint.pth')
    ```

    Using a pre-trained model (automatic, users don't handle config directly):
    ```python
    search = NDFormerMCTS(variables=[x, y])
    search.load_ndformer('hf://YuMeow/ndformer:best.pth')
    # Config is automatically loaded and used for capability validation
    search.fit(X, y)
    ```

    Creating alternative model architectures:
    ```python
    @NDFormerModel.register_model('gcn')
    class GCNNDFormer(NDFormerModel):
        def __init__(self, config, tokenizer):
            super().__init__(config, tokenizer)
            # ... custom architecture ...

    config = NDFormerModelConfig(model='gcn')
    model = NDFormerModel.create(config, tokenizer)
    ```

    ═══════════════════════════════════════════════════════════════════════════
    ATTRIBUTES
    ═══════════════════════════════════════════════════════════════════════════
    """

    # =========================================================================
    # Tokenizer Settings
    # =========================================================================
    # These define how numbers, variables, and operators are converted to
    # token IDs. They are fixed at training time and baked into the checkpoint.
    # =========================================================================
    n_mantissa: int = 4
    """Number of digits in mantissa for number tokenization."""
    min_exponent: int = -100
    """Minimum exponent value for number tokenization."""
    max_exponent: int = 100
    """Maximum exponent value for number tokenization."""
    max_var_num: int = 10
    """Maximum number of variables per nettype (node/edge/scalar)."""

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # These define the transformer and GNN structure. Changing these requires
    # training a new model from scratch.
    # =========================================================================
    model: str = 'default'
    """
    Model architecture type. Used by NDFormerModel.create() to select subclass.

    Available models are registered via @NDFormerModel.register_model('name').
    Default is 'default' (the base NDFormerModel architecture).
    """
    n_head: int = 8
    """Number of attention heads in multi-head self-attention."""
    d_emb: int = 128
    """Dimension of token embeddings and hidden states."""
    d_ff: int = 128 * 4
    """Dimension of feed-forward network intermediate layer."""
    dropout: float = 0.2
    """Dropout probability applied to embeddings and attention."""
    n_GNN_layers: int = 2
    """Number of graph neural network layers for encoding graph topology."""
    n_transformer_encoder_layers: int = 2
    """Number of transformer encoder layers."""
    n_transformer_decoder_layers: int = 2
    """Number of transformer decoder layers for autoregressive generation."""
    use_aux_input: bool = True
    """Whether to use auxiliary inputs (parent/nettype information)."""

    # =========================================================================
    # Model Capabilities
    # =========================================================================
    # These define what the model can handle. NDFormerMCTS validates against
    # these to ensure search stays within model capabilities.
    # =========================================================================
    max_seq_len: int = 100
    """
    Maximum sequence length the model can handle.

    Note: NDFormerMCTS uses this for capability validation - search with
    max_len > max_seq_len may produce unreliable results.
    """
    operands: Tuple[str] = field(default_factory=lambda: (
        'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Max', 'Min',
        'Identity',
        'Sin', 'Cos', 'Tan', 'Sec', 'Csc', 'Cot', 'Arcsin', 'Arccos', 'Arctan',
        'Log', 'LogAbs', 'Exp', 'Abs', 'Neg', 'Inv',
        'Sqrt', 'SqrtAbs', 'Pow2', 'Pow3',
        'Sinh', 'Cosh', 'Tanh', 'Coth', 'Sech', 'Csch',
        'Sigmoid', 'Regular',
        'Sour', 'Targ', 'Aggr', 'Rgga', 'Readout',
    ))
    """
    Tuple of operator class names in the model vocabulary.

    Note: NDFormerMCTS may check if its operator set is compatible with
    the trained model's vocabulary.
    """

    # =========================================================================
    # Training Data Distribution (informational)
    # =========================================================================
    # These define the distribution of synthetic training data. They are
    # stored for reference but NOT used by NDFormerMCTS.
    # =========================================================================
    min_data_num: int = 100
    """Minimum number of samples per training equation."""
    max_data_num: int = 200
    """Maximum number of samples per training equation."""
    min_node_num: int = 10
    """Minimum number of nodes in generated graphs."""
    max_node_num: int = 100
    """Maximum number of nodes in generated graphs."""
    min_edge_num: int = 20
    """Minimum number of edges in generated graphs."""
    max_edge_num: int = 600
    """Maximum number of edges in generated graphs."""
    min_var_val: int = -10
    """Minimum absolute value for variable sampling."""
    max_var_val: int = 10
    """Maximum absolute value for variable sampling."""
    min_coeff_val: int = -20
    """Minimum value for equation coefficients."""
    max_coeff_val: int = 20
    """Maximum value for equation coefficients."""
