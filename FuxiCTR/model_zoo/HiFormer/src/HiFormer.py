# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
A HiFormer-based CTR prediction model for FuxiCTR.

Architecture:
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection (embedding_dim -> ffn_in_dim)
        -> [Optional Positional Embedding]
        -> HiFormer Encoder Layers (Low-rank Multi-head Attention + SwishGLU FFN + Residual)
        -> Mean pooling over tokens
        -> SwishGLU MLP -> Sigmoid

The low-rank attention decomposes the full Q/K/V projection matrices
via A*B factorization to reduce parameters, followed by token-specific
output projection.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


class SwishGLU(nn.Module):
    """
    SwishGLU feed-forward block.
    forward(x) = down_proj( Swish(gate_proj(x)) * up_proj(x) )
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0, use_bias=False):
        super(SwishGLU, self).__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class TokenSpecificSwishGLU(nn.Module):
    """
    Token-specific SwishGLU: each token position has its own projection matrices.
    Accelerated via torch.einsum.
    """
    def __init__(self, seq_len, d_model, d_ff, dropout=0.0):
        super(TokenSpecificSwishGLU, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_ff = d_ff

        self.W_gate = nn.Parameter(torch.empty(seq_len, d_model, d_ff))
        self.W_up = nn.Parameter(torch.empty(seq_len, d_model, d_ff))
        self.W_down = nn.Parameter(torch.empty(seq_len, d_ff, d_model))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_up)
        nn.init.xavier_uniform_(self.W_down)

    def forward(self, x):
        gate = torch.einsum('btd,tdf->btf', x, self.W_gate)
        up = torch.einsum('btd,tdf->btf', x, self.W_up)
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)
        out = torch.einsum('btf,tfd->btd', hidden, self.W_down)
        return out


class HiFormerLowRankAttention(nn.Module):
    """
    Multi-head Low-rank Attention for HiFormer.
    Q/K/V are projected via low-rank matrix factorization A @ B
    to avoid storing a huge [T*D, T*D] matrix.
    """
    def __init__(self, seq_len, d_model, num_heads, low_rank_dim=64, dropout=0.1):
        super(HiFormerLowRankAttention, self).__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.low_rank_dim = low_rank_dim
        self.flat_dim = seq_len * d_model

        # Low-rank Q projection: [T*D, r] and [r, T*D]
        self.A_Q = nn.Parameter(torch.empty(self.flat_dim, low_rank_dim))
        self.B_Q = nn.Parameter(torch.empty(low_rank_dim, self.flat_dim))

        # Low-rank K projection
        self.A_K = nn.Parameter(torch.empty(self.flat_dim, low_rank_dim))
        self.B_K = nn.Parameter(torch.empty(low_rank_dim, self.flat_dim))

        # Low-rank V projection
        self.A_V = nn.Parameter(torch.empty(self.flat_dim, low_rank_dim))
        self.B_V = nn.Parameter(torch.empty(low_rank_dim, self.flat_dim))

        # Token-specific output projection: [T, D, D]
        self.O = nn.Parameter(torch.empty(seq_len, d_model, d_model))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        init_std = 0.001
        for p in [self.A_Q, self.B_Q, self.A_K, self.B_K, self.A_V, self.B_V]:
            # truncated normal initialization (manual fallback for older PyTorch)
            with torch.no_grad():
                torch.nn.init.normal_(p, mean=0.0, std=init_std)
                p.clamp_(min=-2*init_std, max=2*init_std)
        nn.init.xavier_uniform_(self.O)

    def forward(self, X):
        """
        Args:
            X: [B, T, D]
        Returns:
            outputs: [B, T, D]
        """
        B, T, D = X.shape
        flat_dim = T * D

        # Flatten to [B, T*D]
        X_flat = X.reshape(B, flat_dim)

        # Low-rank Q/K/V projections: [B, T*D]
        querys_0 = X_flat @ self.A_Q @ self.B_Q
        keys_0   = X_flat @ self.A_K @ self.B_K
        values_0 = X_flat @ self.A_V @ self.B_V

        # Reshape to multi-head: [B, T, nh, head_dim] -> [nh, B, T, head_dim]
        querys = querys_0.view(B, T, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        keys   = keys_0.view(B, T, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        values = values_0.view(B, T, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(querys, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, values)  # [nh, B, T, head_dim]

        # Concatenate heads: [nh, B, T, head_dim] -> [B, T, D]
        attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(B, T, D)

        # Token-specific output projection
        outputs = torch.einsum('btd,tdo->bto', attn_output, self.O)
        return outputs


class HiFormerEncoderLayer(nn.Module):
    """
    HiFormer Encoder Layer with Low-rank Attention and SwishGLU FFN.
    Post-LayerNorm architecture.
    """
    def __init__(self, seq_len, d_model, num_heads, low_rank_dim=64,
                 dim_feedforward=128, dropout=0.1):
        super(HiFormerEncoderLayer, self).__init__()
        self.self_attn = HiFormerLowRankAttention(
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            low_rank_dim=low_rank_dim,
            dropout=dropout
        )
        self.ffn = TokenSpecificSwishGLU(
            seq_len=seq_len,
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: [B, T, D]
        Returns:
            out: [B, T, D]
        """
        # Low-rank self-attention + residual + layer norm
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # SwishGLU FFN + residual + layer norm
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class HiFormer(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="HiFormer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 ffn_in_dim=None,
                 num_transformer_layers=3,
                 num_heads=4,
                 low_rank_dim=64,
                 transformer_dropout=0.1,
                 ffn_out_dim=128,
                 use_pos_embedding=True,
                 output_mlp_hidden_units=[128, 64],
                 net_dropout=0.0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 double_tokens=False,
                 **kwargs):
        """
        Args:
            feature_map: FeatureMap object from FuxiCTR.
            embedding_dim: Dimension of feature embeddings (now fixed internally to 10).
            ffn_in_dim: Dimension projected to before feeding into encoder layers.
            num_transformer_layers: Number of HiFormer encoder layers.
            num_heads: Number of attention heads.
            low_rank_dim: Rank dimension for low-rank Q/K/V projections.
            transformer_dropout: Dropout rate inside encoder layers and output MLP.
            ffn_out_dim: Hidden dimension of the SwishGLU in encoder layers.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")
        super(HiFormer, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs)

        self.double_tokens = double_tokens
        self.emb_dim = embedding_dim
        if self.double_tokens:
            assert self.emb_dim % 2 == 0, f"emb_dim ({self.emb_dim}) must be even for token splitting"
        self.embedding_layer = FeatureEmbedding(feature_map, self.emb_dim)
        self.use_pos_embedding = use_pos_embedding

        # Embedding projection: embedding_dim -> ffn_in_dim
        if ffn_in_dim is not None and ffn_in_dim != self.emb_dim:
            self.embedding_proj = nn.Linear(self.emb_dim, ffn_in_dim)
            proj_dim = ffn_in_dim
        else:
            self.embedding_proj = None
            proj_dim = self.emb_dim

        self.num_fields = feature_map.num_fields
        self.seq_len = self.num_fields * (2 if self.double_tokens else 1)
        if self.double_tokens:
            assert proj_dim % 2 == 0, f"proj_dim ({proj_dim}) must be even for token splitting"
        self.d_model = proj_dim // 2 if self.double_tokens else proj_dim

        # Positional Embedding: learnable parameter of shape [1, seq_len, ffn_in_dim]
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        # HiFormer Encoder Layers
        self.transformer_encoder = nn.ModuleList([
            HiFormerEncoderLayer(
                seq_len=self.seq_len,
                d_model=self.d_model,
                num_heads=num_heads,
                low_rank_dim=low_rank_dim,
                dim_feedforward=ffn_out_dim,
                dropout=transformer_dropout
            )
            for _ in range(num_transformer_layers)
        ])

        self.output_mlp = nn.Linear(self.d_model, 1)

        # FuxiCTR lifecycle methods
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs: dict containing feature tensors and labels.

        Returns:
            dict: {"y_pred": tensor of shape (batch_size, 1)}
        """
        X = self.get_inputs(inputs)

        # 1. Feature Embedding: (batch_size, num_fields, 10)
        feature_emb = self.embedding_layer(X)  # [B, L, emb_dim]
        if self.embedding_proj is not None:
            feature_emb = self.embedding_proj(feature_emb)  # [B, L, proj_dim]
        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)  # [B, 2L, d_model]
        x = feature_emb

        # 2. Optional: add positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        # 3. HiFormer Encoder
        for layer in self.transformer_encoder:
            x = layer(x)  # (batch, seq_len, ffn_in_dim)

        # 4. Aggregation: mean pooling over tokens
        output = x.mean(dim=1)  # (batch, ffn_in_dim)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
