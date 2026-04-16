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
A FAT (Factorized Attention Transformer) based CTR prediction model for FuxiCTR.

Architecture:
    FeatureEmbedding
        -> [Optional Positional Embedding]
        -> FAT Encoder Layers (Basis-shared token-specific Q/K + Sinkhorn-regularized attention + SwishGLU FFN)
        -> Mean pooling over tokens
        -> SwishGLU MLP -> Sigmoid

The attention mechanism uses shared basis matrices combined with token-specific
meta weights to generate Q/K projections, and a learnable token-specific V
projection. Attention scores are further mixed with a token-token interaction
matrix A_logits and regularized via Sinkhorn normalization.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


def log_sinkhorn(log_alpha, n_iters=20, temperature=1.0):
    """
    Log-domain Sinkhorn normalization for attention regularization.
    Args:
        log_alpha: tensor of arbitrary shape
        n_iters: number of Sinkhorn iterations
        temperature: temperature for Sinkhorn scaling
    Returns:
        normalized tensor of the same shape
    """
    log_alpha = log_alpha / temperature
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


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


class FATAttention(nn.Module):
    """
    Factorized Attention with shared basis weights and Sinkhorn regularization.
    """
    def __init__(self, seq_len, d_model, num_heads, basis_num=8, att_emb_size=None,
                 sinkhorn_iters=20, dropout=0.1):
        super(FATAttention, self).__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.att_emb_size = att_emb_size if att_emb_size is not None else d_model // num_heads
        self.basis_num = basis_num
        self.sinkhorn_iters = sinkhorn_iters

        # Token embeddings for meta-weight generation [T, D]
        self.token_embeddings = nn.Parameter(torch.empty(seq_len, d_model))

        # Meta mappers: token-specific linear layers generating basis weights
        self.Q_meta_maps = nn.ModuleList([
            nn.Linear(d_model, basis_num, bias=False) for _ in range(seq_len)
        ])
        self.K_meta_maps = nn.ModuleList([
            nn.Linear(d_model, basis_num, bias=False) for _ in range(seq_len)
        ])

        # Shared basis matrices [basis_num, D, num_heads * att_emb_size]
        self.Q = nn.Parameter(torch.empty(basis_num, d_model, num_heads * self.att_emb_size))
        self.K = nn.Parameter(torch.empty(basis_num, d_model, num_heads * self.att_emb_size))

        # Token-specific V projection [T, D, nh * att_emb_size] (normal linear transform, no Sinkhorn)
        self.W_V = nn.Parameter(torch.empty(seq_len, d_model, num_heads * self.att_emb_size))

        # Token-token interaction and gating matrices
        self.A_logits = nn.Parameter(torch.empty(seq_len, seq_len))
        self.lambda_logits = nn.Parameter(torch.empty(seq_len, seq_len))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.token_embeddings)
        for m in self.Q_meta_maps:
            nn.init.xavier_uniform_(m.weight)
        for m in self.K_meta_maps:
            nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.A_logits)
        nn.init.xavier_uniform_(self.lambda_logits)

    def forward(self, X):
        """
        Args:
            X: [B, T, D]
        Returns:
            outputs: [B, T, D]
        """
        B, T, D = X.shape

        # Generate token-specific basis weights [T, basis_num]
        q_weights = torch.stack([self.Q_meta_maps[i](self.token_embeddings[i]) for i in range(T)])
        k_weights = torch.stack([self.K_meta_maps[i](self.token_embeddings[i]) for i in range(T)])

        # Generate token-specific Q/K via basis combination [T, D, att_emb_size]
        Q_tilde = torch.einsum('tk,kde->tde', q_weights, self.Q)
        K_tilde = torch.einsum('tk,kde->tde', k_weights, self.K)

        # Token-specific projections
        querys = torch.einsum('btd,tde->bte', X, Q_tilde)   # [B, T, att_emb_size]
        keys   = torch.einsum('btd,tde->bte', X, K_tilde)   # [B, T, att_emb_size]
        values = torch.einsum('btd,tdn->btn', X, self.W_V)  # [B, T, nh * att_emb_size]

        # Multi-head reshape: [B, T, nh, head_dim] -> [nh, B, T, head_dim]
        querys = querys.view(B, T, self.num_heads, self.att_emb_size).permute(2, 0, 1, 3)
        keys   = keys.view(B, T, self.num_heads, self.att_emb_size).permute(2, 0, 1, 3)
        values = values.view(B, T, self.num_heads, self.att_emb_size).permute(2, 0, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(querys, keys.transpose(-2, -1)) / math.sqrt(self.att_emb_size)
        att_scores = F.softmax(scores, dim=-1)
        att_scores = self.dropout(att_scores)

        # Mix with A_logits and lambda gating
        lambda_weight = torch.sigmoid(self.lambda_logits)   # [T, T]
        A_logits_sym = (self.A_logits + self.A_logits.transpose(0, 1)) * 0.5

        # Expand to [1, 1, T, T] for broadcasting with [nh, B, T, T]
        A_expand = A_logits_sym.unsqueeze(0).unsqueeze(0)           # [1, 1, T, T]
        lambda_expand = lambda_weight.unsqueeze(0).unsqueeze(0)     # [1, 1, T, T]

        att_scores = lambda_expand * A_expand + (1 - lambda_expand) * att_scores

        # Symmetrize and Sinkhorn normalize
        att_scores = (att_scores + att_scores.transpose(-2, -1)) * 0.5
        att_scores = log_sinkhorn(att_scores, n_iters=self.sinkhorn_iters, temperature=1.0)

        # Apply attention to values
        result = torch.matmul(att_scores, values)  # [nh, B, T, D]

        # Merge heads back to [B, T, D]
        result = result.permute(1, 2, 0, 3).contiguous().view(B, T, D)
        return result


class FATEncoderLayer(nn.Module):
    """
    FAT Encoder Layer with FAT Attention and SwishGLU FFN.
    Post-LayerNorm architecture.
    """
    def __init__(self, seq_len, d_model, num_heads, basis_num=8, att_emb_size=None,
                 sinkhorn_iters=20, dim_feedforward=128, dropout=0.1):
        super(FATEncoderLayer, self).__init__()
        self.self_attn = FATAttention(
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            basis_num=basis_num,
            att_emb_size=att_emb_size,
            sinkhorn_iters=sinkhorn_iters,
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
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class FAT(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="FAT",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_transformer_layers=2,
                 num_heads=4,
                 basis_num=8,
                 att_emb_size=None,
                 sinkhorn_iters=20,
                 transformer_dropout=0.1,
                 ffn_dim=128,
                 use_pos_embedding=True,
                 output_mlp_hidden_units=[128, 64],
                 net_dropout=0.0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        """
        Args:
            feature_map: FeatureMap object from FuxiCTR.
            embedding_dim: Dimension of feature embeddings.
            num_transformer_layers: Number of FAT encoder layers.
            num_heads: Number of attention heads.
            basis_num: Number of shared basis matrices for Q/K generation.
            att_emb_size: Dimension of each attention head. Defaults to embedding_dim // num_heads.
            sinkhorn_iters: Number of Sinkhorn iterations for attention regularization.
            transformer_dropout: Dropout rate inside encoder layers and output MLP.
            ffn_dim: Hidden dimension of the SwishGLU in encoder layers.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        super(FAT, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.use_pos_embedding = use_pos_embedding

        self.num_fields = feature_map.num_fields
        self.seq_len = self.num_fields

        # Positional Embedding
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embedding_dim))

        # FAT Encoder Layers
        self.transformer_encoder = nn.ModuleList([
            FATEncoderLayer(
                seq_len=self.seq_len,
                d_model=embedding_dim,
                num_heads=num_heads,
                basis_num=basis_num,
                att_emb_size=att_emb_size,
                sinkhorn_iters=sinkhorn_iters,
                dim_feedforward=ffn_dim,
                dropout=transformer_dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # Output MLP (SwishGLU-based)
        mlp_input_dim = embedding_dim

        if not isinstance(net_dropout, list):
            dropout_rates = [net_dropout] * len(output_mlp_hidden_units)
        else:
            dropout_rates = net_dropout

        output_layers = []
        prev_dim = mlp_input_dim
        for idx, h in enumerate(output_mlp_hidden_units):
            output_layers.append(SwishGLU(
                input_dim=prev_dim,
                hidden_dim=h,
                output_dim=h,
                dropout=dropout_rates[idx]
            ))
            if batch_norm:
                output_layers.append(nn.BatchNorm1d(h))
            prev_dim = h
        output_layers.append(nn.Linear(prev_dim, 1))
        self.output_mlp = nn.Sequential(*output_layers)

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

        # 1. Feature Embedding
        feature_emb = self.embedding_layer(X)  # [B, L, D]
        x = feature_emb

        # 2. Optional positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        # 3. FAT Encoder
        for layer in self.transformer_encoder:
            x = layer(x)  # (batch, seq_len, emb_dim)

        # 4. Aggregation: mean pooling over tokens
        output = x.mean(dim=1)  # (batch, emb_dim)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
