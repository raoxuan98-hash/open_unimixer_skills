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
A RankMixer-based CTR prediction model for FuxiCTR.

Architecture:
    FeatureEmbedding
        -> [Optional Positional Embedding]
        -> RankMixer Layers (Rule-based mixing + Token-specific SwishGLU + LayerNorm)
        -> Mean pooling over tokens
        -> SwishGLU MLP -> Sigmoid

The rule-based mixing reshapes hidden states as [B, L, L, D//L] and transposes
to achieve explicit feature crossing, followed by token-specific SwishGLU FFN.
"""

import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


class SwishGLU(nn.Module):
    """
    Standard SwishGLU feed-forward block.
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

        # Token-specific weights: [L, D, d_ff] and [L, d_ff, D]
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
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        # Token-specific projections via einsum
        gate = torch.einsum('bld,ldf->blf', x, self.W_gate)
        up = torch.einsum('bld,ldf->blf', x, self.W_up)
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)
        out = torch.einsum('blf,lfd->bld', hidden, self.W_down)
        return out


class RankMixerLayer(nn.Module):
    """
    RankMixer encoder layer:
        1. Rule-based mixing (reshape + transpose)
        2. Residual + LayerNorm
        3. Token-specific SwishGLU
        4. Residual + LayerNorm
    """
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1):
        super(RankMixerLayer, self).__init__()
        assert d_model % seq_len == 0, \
            f"d_model ({d_model}) must be divisible by seq_len ({seq_len}) for rule-based mixing"
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_ffn = TokenSpecificSwishGLU(seq_len, d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        B, L, D = src.shape

        # Rule-based mixing: [B, L, D] -> [B, L, L, D//L] -> transpose -> [B, L, D]
        mixing_h = src.view(B, L, L, D // L)
        mixing_h = mixing_h.transpose(1, 2).contiguous()
        mixing_h = mixing_h.view(B, L, D)

        mixed = self.norm1(src + self.dropout1(mixing_h))

        # Token-specific SwishGLU
        ffn_out = self.token_ffn(mixed)
        out = self.norm2(mixed + self.dropout2(ffn_out))
        return out


class RankMixer(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="RankMixer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_transformer_layers=2,
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
                           Must be divisible by num_fields.
            num_transformer_layers: Number of RankMixer layers.
            transformer_dropout: Dropout rate inside RankMixer layers and output MLP.
            ffn_dim: Hidden dimension of the token-specific SwishGLU.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        super(RankMixer, self).__init__(
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

        assert embedding_dim % self.seq_len == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_fields ({self.num_fields})"

        # Positional Embedding: learnable parameter of shape [1, seq_len, embedding_dim]
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embedding_dim))

        # RankMixer Layers
        self.transformer_encoder = nn.ModuleList([
            RankMixerLayer(
                seq_len=self.seq_len,
                d_model=embedding_dim,
                d_ff=ffn_dim,
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

        # 1. Feature Embedding: (batch_size, num_fields, embedding_dim)
        feature_emb = self.embedding_layer(X)  # [B, L, D]
        x = feature_emb

        # 2. Optional: add positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        # 3. RankMixer Encoder
        for layer in self.transformer_encoder:
            x = layer(x)  # (batch, seq_len, emb_dim)

        # 4. Aggregation: mean pooling over tokens
        output = x.mean(dim=1)  # (batch, emb_dim)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
