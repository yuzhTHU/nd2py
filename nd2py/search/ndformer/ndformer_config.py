# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class NDFormerConfig:
    # Tokenizer
    n_mantissa: int = 4
    min_exponent: int = -100
    max_exponent: int = 100
    max_var_num: int = 10
    # Model
    n_head: int = 8
    d_emb: int = 128
    d_ff: int = 128*4
    dropout: float = 0.1
    n_GNN_layers: int = 2
    n_transformer_encoder_layers: int = 2
    n_transformer_decoder_layers: int = 2
    use_aux_input: bool = True
    # Eq Generator
    max_seq_len: int = 100  # no less than GDExpr.max_complexity + 3
    operands: Tuple[str] = (
        'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Max', 'Min',
        'Identity',
        'Sin', 'Cos', 'Tan', 'Sec', 'Csc', 'Cot', 'Arcsin', 'Arccos', 'Arctan',
        'Log', 'LogAbs', 'Exp', 'Abs', 'Neg', 'Inv',
        'Sqrt', 'SqrtAbs', 'Pow2', 'Pow3',
        'Sinh', 'Cosh', 'Tanh', 'Coth', 'Sech', 'Csch',
        'Sigmoid', 'Regular',
        'Sour', 'Targ', 'Aggr', 'Rgga', 'Readout',
    )
    # Data Generator
    min_data_num: int = 100
    max_data_num: int = 200
    min_node_num: int = 10
    max_node_num: int = 100
    min_edge_num: int = 20
    max_edge_num: int = 600
    min_var_val: int = -10
    max_var_val: int = 10
    min_coeff_val: int = -20
    max_coeff_val: int = 20
