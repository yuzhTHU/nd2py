from dataclasses import dataclass

@dataclass
class NDformerConfig:
    d_emb: int = 512
    dropout: float = 0.1
    d_data_feat: int = 16 # sign bit (1) + exponent bits (5) + mantissa bits (10)
    n_node_vars: int = 6
    n_edge_vars: int = 6
    n_transformer_encoder_layers: int = 2
    n_GNN_layers: int = 2
    max_sample_num: int = 3000
    split: bool = False
    use_aux_input: bool = True
    n_words: int = 60 # vocabulary size
    n_transformer_decoder_layers: int = 2
    max_seq_len: int = 100  # no less than GDExpr.max_complexity + 3
