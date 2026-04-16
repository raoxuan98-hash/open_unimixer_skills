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
A UniMixer-Lite based CTR prediction model for FuxiCTR.

Architecture:
    FeatureEmbedding
        -> [Optional Positional Embedding]
        -> Siamese dual-branch (x and y)
        -> UniMixer-Lite Encoder Layers (Kronecker doubly-stochastic mixing + Token-specific SwishGLU + LayerNorm)
        -> Mean pooling over tokens from main branch x
        -> SwishGLU MLP -> Sigmoid

The Kronecker mixing factorizes global token permutation (A) and local
chunk-specific permutation (W) via low-rank decomposition and basis
combination, both regularized by Sinkhorn normalization.
"""

import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


def log_sinkhorn(log_alpha, n_iters=20, temperature=1.0):
    """
    Log-domain Sinkhorn normalization.
    Operates on the last two dimensions.
    """
    log_alpha = log_alpha / temperature
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


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


class KroneckerMixer(nn.Module):
    """
    Kronecker interaction with doubly-stochastic constraints via Sinkhorn.
    Input: [B, total_dim] where total_dim = seq_len * d_model
    Global permutation A uses full [N, N] parameters;
    local permutation W uses basis combination (parameter sharing).
    """
    def __init__(self, total_dim, block_size, basis_num=8, sinkhorn_iters=20, sinkhorn_temperature=0.2, dropout=0.0):
        super(KroneckerMixer, self).__init__()
        assert total_dim % block_size == 0, \
            f"total_dim ({total_dim}) must be divisible by block_size ({block_size})"
        self.N = total_dim // block_size
        self.K = block_size
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temperature = sinkhorn_temperature

        # Global interaction matrix A: full [N, N]
        self.A_logits = nn.Parameter(torch.empty(self.N, self.N))

        # Local basis W: [N, basis_num] @ [basis_num, K, K]
        self.W_1 = nn.Parameter(torch.empty(self.N, basis_num))
        self.W_V = nn.Parameter(torch.empty(basis_num, self.K, self.K))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.A_logits)
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, x):
        """
        Args:
            x: [B, total_dim]
        Returns:
            out: [B, total_dim]
        """
        B = x.shape[0]
        N, K = self.N, self.K

        A = log_sinkhorn(self.A_logits.unsqueeze(0), n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temperature).squeeze(0)

        W_logits = torch.einsum('tk,kde->tde', self.W_1, self.W_V)
        W = log_sinkhorn(W_logits.unsqueeze(0), n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temperature).squeeze(0)

        x_reshaped = x.view(B, N, K)
        x_local = torch.einsum('bni,nio->bno', x_reshaped, W)
        x_global = torch.einsum('mn,bno->bmo', A, x_local)
        output = x_global.reshape(B, N * K)
        return output


class UniMixerLiteLayer(nn.Module):
    """
    UniMixer-Lite Encoder Layer with siamese dual-branch and Kronecker mixing.
    """
    def __init__(self, seq_len, d_model, block_size, basis_num, d_ff,
                 sinkhorn_iters=20, sinkhorn_temperature=0.2, dropout=0.1):
        super(UniMixerLiteLayer, self).__init__()
        total_dim = seq_len * d_model
        self.mixer = KroneckerMixer(
            total_dim=total_dim,
            block_size=block_size,
            basis_num=basis_num,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_temperature=sinkhorn_temperature,
            dropout=dropout
        )
        self.ffn = TokenSpecificSwishGLU(seq_len, d_model, d_ff, dropout)

        self.norm_y = nn.LayerNorm(d_model)
        self.norm_mix = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        Args:
            x: [B, T, D]
            y: [B, T, D]
        Returns:
            final_x: [B, T, D]
            final_y: [B, T, D]
        """
        B, T, D = x.shape

        # Siamese mixing input
        normed_y = self.norm_y(y)
        mixing_input = x + normed_y

        # Kronecker mixing: [B, T, D] -> [B, T*D] -> mix -> [B, T*D] -> [B, T, D]
        mixing_h = mixing_input.reshape(B, T * D)
        mixing_h = self.mixer(mixing_h)
        mixing_h = mixing_h.view(B, T, D)

        pffn_input = self.norm_mix(mixing_input + self.dropout(mixing_h))

        # Token-specific SwishGLU
        ffn_output = self.ffn(pffn_input)

        final_x = self.norm_x(x + self.dropout(ffn_output))
        final_y = y + self.dropout(ffn_output)

        return final_x, final_y


class UniMixerLite(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="UniMixerLite",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 num_transformer_layers=2,
                 block_size=4,
                 basis_num=8,
                 sinkhorn_iters=20,
                 sinkhorn_temperature=0.2,
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
            num_transformer_layers: Number of UniMixer-Lite encoder layers.
            block_size: Chunk size for Kronecker mixing. Defaults to 4.
            basis_num: Number of basis matrices for local interaction W.
            sinkhorn_iters: Number of Sinkhorn iterations.
            sinkhorn_temperature: Temperature for Sinkhorn normalization. Defaults to 0.2.
            transformer_dropout: Dropout rate inside encoder layers and output MLP.
            ffn_dim: Hidden dimension of the token-specific SwishGLU.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        super(UniMixerLite, self).__init__(
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

        assert (self.seq_len * embedding_dim) % block_size == 0, \
            f"seq_len * embedding_dim ({self.seq_len * embedding_dim}) must be divisible by block_size ({block_size})"
        self.block_size = block_size

        # Siamese y-branch projection
        self.y_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Positional Embedding
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embedding_dim))

        # UniMixer-Lite Encoder Layers
        self.transformer_encoder = nn.ModuleList([
            UniMixerLiteLayer(
                seq_len=self.seq_len,
                d_model=embedding_dim,
                block_size=block_size,
                basis_num=basis_num,
                d_ff=ffn_dim,
                sinkhorn_iters=sinkhorn_iters,
                sinkhorn_temperature=sinkhorn_temperature,
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
        y = self.y_proj(feature_emb)

        # 2. Optional positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding
            y = y + self.pos_embedding

        # 3. UniMixer-Lite Encoder
        for layer in self.transformer_encoder:
            x, y = layer(x, y)

        # 4. Aggregation: use main branch x, mean pooling over tokens
        output = x.mean(dim=1)  # (batch, emb_dim)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
