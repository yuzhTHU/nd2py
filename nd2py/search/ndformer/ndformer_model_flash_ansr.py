# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""
FLASH-ANSR variant of NDFormer.

Based on the paper describing FLASH-ANSR architecture:
- Pre-norm Transformer (norm_first=True)
- Set Transformer encoder with induction points
- FlashAttention support (via torch.nn.MultiheadAttention with backend selection)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .ndformer_model import NDFormerModel
from .ndformer_config import NDFormerModelConfig
from .ndformer_tokenizer import NDFormerTokenizer


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer Encoder with induction points.

    Drop-in replacement for nn.TransformerEncoder with identical forward signature.
    Induction points are used internally but output shape matches input shape.

    Architecture (Lee et al., 2019):
    1. Induction points attend to input data (cross-attention)
    2. Self-attention among induction points
    3. Induction points attend back to original positions (output projection)

    Args:
        encoder_layer: Not used (kept for API compatibility)
        num_layers: Number of transformer layers
        norm: Final normalization layer
        d_model: Embedding dimension
        n_induction_points: Number of learnable induction points
    """

    def __init__(
        self,
        encoder_layer=None,
        num_layers=2,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
        d_model: int = None,
        n_induction_points: int = 128,
        n_head: int = 8,
    ):
        super().__init__()
        self.n_induction_points = n_induction_points
        self.induction_points = nn.Parameter(torch.randn(1, n_induction_points, d_model))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=norm,
            enable_nested_tensor=enable_nested_tensor,
            mask_check=mask_check
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            batch_first=True,
        )

    def forward(
        self,
        src: torch.Tensor,
        mask=None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal=None,
    ) -> torch.Tensor:
        """
        Args:
            src: Input tensor (batch, seq_len, d_model) - GNN encoded nodes
            mask: Not used (kept for API compatibility)
            src_key_padding_mask: Padding mask (batch, seq_len), True for padding
            is_causal: Not used (kept for API compatibility)

        Returns:
            Output tensor with same shape as input (batch, seq_len, d_model)
        """
        ind_points = self.induction_points.expand(src.shape[0], -1, -1)  # (batch, I, d_model)
        combined_src = torch.cat([ind_points, src], dim=1)  # (batch, I+seq_len, d_model)
        if src_key_padding_mask is not None:
            # Pad induction points as False (valid), keep original padding
            ind_pad = torch.zeros(
                *ind_points.shape[:2],
                device=src_key_padding_mask.device,
                dtype=src_key_padding_mask.dtype
            )
            combined_src_key_padding_mask = torch.cat([ind_pad, src_key_padding_mask], dim=1)
        else:
            combined_src_key_padding_mask = None
        out = self.transformer(combined_src, src_key_padding_mask=combined_src_key_padding_mask)
        ind_out = out[:, :self.n_induction_points, :]  # (batch, I, d_model)
        attn_output, _ = self.attention(query=src, key=ind_out, value=ind_out) # (batch, seq_len, d_model)
        attn_output[src_key_padding_mask, :] = 0.0
        return attn_output


@NDFormerModel.register_model('flash_ansr')
class FlashANSRNDFormer(NDFormerModel):
    """
    FLASH-ANSR: Transformer-based symbolic regression with Set Transformer encoder
    and pre-norm architecture.

    Key features:
    - Pre-norm Transformer (norm_first=True)
    - Set Transformer encoder with learnable induction points
    - LayerNorm for normalization

    Reuses NDFormerModel.encode_graph() and NDFormerModel.decode_sequence().
    """

    def __init__(self, config: NDFormerModelConfig, tokenizer: NDFormerTokenizer):
        super().__init__(config, tokenizer)

        # Replace encoder with Set Transformer variant (induction points)
        # Decoder uses standard nn.TransformerDecoder with norm_first=True
        self.n_induction_points = getattr(config, 'n_induction_points', 128)
        self.transformer_encoder = SetTransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.d_emb,
                nhead=self.config.n_head,
                dim_feedforward=self.config.d_ff,
                dropout=self.config.dropout,
                batch_first=True,
                # norm_first=True,
            ),
            num_layers=self.config.n_transformer_encoder_layers,
            # norm=nn.LayerNorm(self.config.d_emb),
            d_model=self.config.d_emb,
            n_induction_points=self.n_induction_points,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.config.d_emb,
                nhead=self.config.n_head,
                dim_feedforward=self.config.d_ff,
                dropout=self.config.dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.config.n_transformer_decoder_layers,
            norm=nn.LayerNorm(self.config.d_emb),
        )
