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
UniMixerLayerNorm: A TokenMixer-Large variant using nn.LayerNorm instead of
spherical L2 normalization, while keeping sigmoid-gated residual updates.

Architecture:
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection (embedding_dim -> ffn_in_dim)
        -> [Optional Positional Embedding]
        -> TokenMixer-Large Encoder Layers (Dual per-token SwishGLU with
           LayerNorm pre-normalization and LayerNorm-based gated residuals)
        -> Mean pooling over tokens
        -> Linear -> Sigmoid
"""

import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding

from .layer_norm_ops import (
    PerTokenSwishGLU,
    PerTokenSwishGLU_Basis,
    LayerNormResidualUpdate,
)


class SwishGLU(nn.Module):
    """
    Standard SwishGLU feed-forward block (kept for reference / output MLP use).
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
    Accelerated via torch.einsum.  (Original non-spherical version kept for reference)
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
        gate = torch.einsum('bld,ldf->blf', x, self.W_gate)
        up = torch.einsum('bld,ldf->blf', x, self.W_up)
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)
        out = torch.einsum('blf,lfd->bld', hidden, self.W_down)
        return out


class UniMixerLayerNormLayer(nn.Module):
    """
    TokenMixer-Large Encoder Layer with LayerNorm normalization.

    1. Sequence-mixer: apply PerTokenSwishGLU on the sequence dimension
       via transpose, i.e. treat feature dim as token positions.

    2. Feature-mixer: PerTokenSwishGLU on the feature (token) dimension.

    Each sub-layer uses:
        - Pre-LayerNorm
        - Per-token SwishGLU mixing
        - LayerNormResidualUpdate (sigmoid-gated alpha + LayerNorm)
    """
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1,
                 use_basis=False, basis_num=8):
        super(UniMixerLayerNormLayer, self).__init__()

        self.seq_norm = nn.LayerNorm(seq_len)
        self.feat_norm = nn.LayerNorm(d_model)

        if use_basis:
            self.seq_mixer = PerTokenSwishGLU_Basis(
                num_tokens=d_model,
                in_dim=seq_len,
                hidden_dim=seq_len * 10,
                basis_num=basis_num,
                dropout=dropout,
            )
            self.feature_mixer = PerTokenSwishGLU_Basis(
                num_tokens=seq_len,
                in_dim=d_model,
                hidden_dim=d_ff,
                basis_num=basis_num,
                dropout=dropout,
            )
        else:
            self.seq_mixer = PerTokenSwishGLU(
                num_tokens=d_model,
                in_dim=seq_len,
                hidden_dim=seq_len * 10,
                dropout=dropout,
            )
            self.feature_mixer = PerTokenSwishGLU(
                num_tokens=seq_len,
                in_dim=d_model,
                hidden_dim=d_ff,
                dropout=dropout,
            )

        self.seq_residual = LayerNormResidualUpdate(
            num_tokens=seq_len,
            in_dim=d_model,
            alpha_init=0.05,
            gate_type="token_wise",
        )
        self.feature_residual = LayerNormResidualUpdate(
            num_tokens=seq_len,
            in_dim=d_model,
            alpha_init=0.05,
            gate_type="token_wise",
        )

    def forward(self, src):
        """
        Args:
            src: [B, T, D]
        Returns:
            out: [B, T, D]
        """
        # ---- Sub-layer 1: sequence-mixer ----
        # src2 = self.seq_norm(src)          # [B, T, D]
        src2 = src.transpose(1, 2)        # [B, D, T]
        src2 = self.seq_norm(src2)          # [B, D, T]
        src2 = self.seq_mixer(src2)        # [B, D, T]
        src2 = src2.transpose(1, 2)        # [B, T, D]
        src = self.seq_residual(src, src2) # sigmoid-gated residual + LayerNorm

        # ---- Sub-layer 2: feature-mixer ----
        src2 = self.feat_norm(src)         # [B, T, D]
        src2 = self.feature_mixer(src2)    # [B, T, D]
        src = self.feature_residual(src, src2) # sigmoid-gated residual + LayerNorm

        return src


class UniMixerLayerNorm(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="UniMixerLayerNorm",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 ffn_in_dim=None,
                 num_transformer_layers=3,
                 transformer_dropout=0.1,
                 ffn_out_dim=128,
                 use_pos_embedding=True,
                 output_mlp_hidden_units=[128, 64],
                 net_dropout=0.0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 double_tokens=False,
                 use_basis=False,
                 basis_num=8,
                 **kwargs):
        """
        Args:
            feature_map: FeatureMap object from FuxiCTR.
            embedding_dim: Dimension of feature embeddings.
            ffn_in_dim: Dimension projected to before feeding into encoder layers.
            num_transformer_layers: Number of encoder layers.
            transformer_dropout: Dropout rate inside encoder layers and output MLP.
            ffn_out_dim: Hidden dimension of the per-token SwishGLU.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
            double_tokens: If True, split each embedding into two tokens.
            use_basis: If True, use basis-weighted SwishGLU (parameter-efficient).
            basis_num: Number of basis matrices when use_basis=True.
        """
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")
        super(UniMixerLayerNorm, self).__init__(
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

        # LayerNorm after embedding
        self.emb_norm = nn.LayerNorm(self.emb_dim)

        # Positional Embedding
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        # LayerNorm before encoder input
        self.encoder_input_norm = nn.LayerNorm(self.d_model)

        # Encoder Layers (LayerNorm version)
        self.transformer_encoder = nn.ModuleList([
            UniMixerLayerNormLayer(
                seq_len=self.seq_len,
                d_model=self.d_model,
                d_ff=ffn_out_dim,
                dropout=transformer_dropout,
                use_basis=use_basis,
                basis_num=basis_num,
            )
            for _ in range(num_transformer_layers)
        ])
        self.output_norm = nn.LayerNorm(self.d_model)
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

        # 1. Feature Embedding
        feature_emb = self.embedding_layer(X)  # [B, L, emb_dim]
        feature_emb = self.emb_norm(feature_emb)  # LayerNorm

        if self.embedding_proj is not None:
            feature_emb = self.embedding_proj(feature_emb)  # [B, L, proj_dim]

        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)  # [B, 2L, d_model]

        x = feature_emb

        # 2. Optional positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        x = self.encoder_input_norm(x)  # LayerNorm

        # 3. TokenMixer-Large Encoder
        for layer in self.transformer_encoder:
            x = layer(x)  # (batch, seq_len, d_model)

        # 4. Aggregation: mean pooling over tokens
        x = self.output_norm(x)
        output = x.mean(dim=1)  # (batch, d_model)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}

