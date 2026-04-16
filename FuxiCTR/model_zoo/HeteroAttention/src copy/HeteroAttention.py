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
A Heterogeneous Self-Attention based CTR prediction model for FuxiCTR.

Architecture:
    FeatureEmbedding
        -> [Optional CLS Token] + [Optional Positional Embedding]
        -> HeteroTransformerEncoder (Token-specific MultiHeadAttention + SwishGLU FFN + Residual)
        -> [CLS token output OR Flatten all tokens]
        -> SwishGLU MLP -> Sigmoid

In the heterogeneous attention, each token (field) has its own projection
matrices W_q, W_k, W_v and output matrix W_o, which is suitable for
handling heterogeneous features in recommendation systems.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


class HeteroSelfAttention(nn.Module):
    """
    Multi-head Heterogeneous Self-Attention.
    Each token position has token-specific Q/K/V/O projection matrices.
    """
    def __init__(self, seq_len, d_model, num_heads, att_emb_size=None, dropout=0.1):
        super(HeteroSelfAttention, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.att_emb_size = att_emb_size if att_emb_size is not None else d_model // num_heads

        # Token-specific projection matrices
        self.Q = nn.Parameter(torch.empty(seq_len, d_model, num_heads * self.att_emb_size))
        self.K = nn.Parameter(torch.empty(seq_len, d_model, num_heads * self.att_emb_size))
        self.V = nn.Parameter(torch.empty(seq_len, d_model, num_heads * d_model))
        self.O = nn.Parameter(torch.empty(seq_len, num_heads * d_model, d_model))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.O)

    def forward(self, X):
        """
        Args:
            X: [B, T, D]
        Returns:
            outputs: [B, T, D]
        """
        B, T, D = X.shape

        # Token-specific projections via einsum
        querys = torch.einsum('btd,tde->bte', X, self.Q)   # [B, T, nh * att_emb_size]
        keys   = torch.einsum('btd,tde->bte', X, self.K)   # [B, T, nh * att_emb_size]
        values = torch.einsum('btd,tdn->btn', X, self.V)   # [B, T, nh * D]

        # Reshape to multi-head: [B, T, nh, *] -> [nh, B, T, *]
        querys = querys.view(B, T, self.num_heads, self.att_emb_size).permute(2, 0, 1, 3)
        keys   = keys.view(B, T, self.num_heads, self.att_emb_size).permute(2, 0, 1, 3)
        values = values.view(B, T, self.num_heads, D).permute(2, 0, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(querys, keys.transpose(-2, -1)) / math.sqrt(self.att_emb_size)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, values)  # [nh, B, T, D]

        # Concatenate heads
        out = out.permute(1, 2, 0, 3).contiguous().view(B, T, self.num_heads * D)  # [B, T, nh*D]

        # Token-specific output projection
        outputs = torch.einsum('btn,tnd->btd', out, self.O)  # [B, T, D]
        return outputs


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


class HeteroTransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Heterogeneous Self-Attention and SwishGLU FFN.
    Post-LayerNorm architecture (same as PyTorch native default).
    """
    def __init__(self, seq_len, d_model, num_heads, att_emb_size=None,
                 dim_feedforward=128, dropout=0.1):
        super(HeteroTransformerEncoderLayer, self).__init__()
        self.self_attn = HeteroSelfAttention(
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            att_emb_size=att_emb_size,
            dropout=dropout
        )
        self.ffn = SwishGLU(
            input_dim=d_model,
            hidden_dim=dim_feedforward,
            output_dim=d_model,
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
        # Heterogeneous self-attention + residual + layer norm
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # SwishGLU FFN + residual + layer norm
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class HeteroAttention(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="HeteroAttention",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_transformer_layers=2,
                 num_heads=4,
                 att_emb_size=None,
                 transformer_dropout=0.1,
                 ffn_dim=128,
                 use_cls_token=True,
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
            num_transformer_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            att_emb_size: Dimension of each attention head for Q/K projections.
                          If None, defaults to embedding_dim // num_heads.
            transformer_dropout: Dropout rate inside Transformer layers.
            ffn_dim: Hidden dimension of the Feed-Forward Network in Transformer.
            use_cls_token: If True, prepend a learnable CLS token and use its output
                           for final prediction (similar to BERT).
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        super(HeteroAttention, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs)

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.use_cls_token = use_cls_token
        self.use_pos_embedding = use_pos_embedding

        self.num_fields = feature_map.num_fields
        self.seq_len = self.num_fields + (1 if use_cls_token else 0)

        # CLS token: learnable parameter of shape [1, 1, embedding_dim]
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Positional Embedding: learnable parameter of shape [1, seq_len, embedding_dim]
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embedding_dim))

        # Heterogeneous Transformer Encoder Layers
        self.transformer_encoder = nn.ModuleList([
            HeteroTransformerEncoderLayer(
                seq_len=self.seq_len,
                d_model=embedding_dim,
                num_heads=num_heads,
                att_emb_size=att_emb_size,
                dim_feedforward=ffn_dim,
                dropout=transformer_dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # Output MLP (SwishGLU-based)
        if self.use_cls_token:
            mlp_input_dim = embedding_dim
        else:
            mlp_input_dim = self.num_fields * embedding_dim

        # Build SwishGLU output MLP
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

        # 1. Feature Embedding: (batch_size, num_fields, embedding_dim)
        feature_emb = self.embedding_layer(X)

        # 2. Optional: prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(feature_emb.size(0), -1, -1)
            x = torch.cat([cls_tokens, feature_emb], dim=1)  # (batch, seq_len, emb_dim)
        else:
            x = feature_emb

        # 3. Optional: add positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        # 4. Transformer Encoder
        for layer in self.transformer_encoder:
            x = layer(x)  # (batch, seq_len, emb_dim)

        # 5. Aggregation strategy
        if self.use_cls_token:
            # Use the output corresponding to the CLS token
            output = x[:, 0, :]  # (batch, emb_dim)
        else:
            # Flatten all feature-token outputs
            output = x.flatten(start_dim=1)  # (batch, num_fields * emb_dim)

        # 6. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
