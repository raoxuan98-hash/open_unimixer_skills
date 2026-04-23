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
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection (embedding_dim -> ffn_in_dim)
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
    def __init__(self, total_dim, block_size, basis_num=8, sinkhorn_iters=1, sinkhorn_temperature=0.2, dropout=0.0):
        super(KroneckerMixer, self).__init__()
        assert total_dim % block_size == 0, \
            f"total_dim ({total_dim}) must be divisible by block_size ({block_size})"
        self.N = total_dim // block_size
        self.K = block_size
        self.total_dim = total_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temperature = sinkhorn_temperature

        # Global interaction matrix A: full [N, N]
        self.A_logits = nn.Parameter(torch.empty(self.N, self.N))

        # Local basis W: [N, basis_num] @ [basis_num, K, K]
        self.W_1 = nn.Parameter(torch.empty(self.N, basis_num))
        self.W_V = nn.Parameter(torch.empty(basis_num, self.K, self.K))
        
        self.norm1 = nn.LayerNorm(self.K)
        self.norm2 = nn.LayerNorm(self.N)

        self.dropout = nn.Dropout(dropout)
        
        # 初始化随机打乱层：创建固定随机排列
        self.register_buffer('shuffle_idx', torch.randperm(total_dim))
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.A_logits, std=0.01)
        nn.init.normal_(self.W_1, std=0.01)
        nn.init.normal_(self.W_V, std=0.01)

    def forward(self, x):
        """
        Args:
            x: [B, total_dim]
        Returns:
            out: [B, total_dim]
        """
        B = x.shape[0]
        N, K = self.N, self.K
        x_reshaped = x.view(B, N, K)
        A = log_sinkhorn(self.A_logits.unsqueeze(0), n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temperature).squeeze(0)
        W_logits = torch.einsum('tk,kde->tde', self.W_1, self.W_V)
        W = log_sinkhorn(W_logits.unsqueeze(0), n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temperature).squeeze(0)
        x_local = torch.einsum('bni,nio->bno', x_reshaped, W)
        x_local = self.norm2(x_local.transpose(1, 2))
        x_global = x_local @ A
        x_global = x_global.transpose(1, 2)
        output =  x_global.reshape(B, N * K)
        return output


class UniMixerLiteLayer(nn.Module):
    """
    UniMixer-Lite Encoder Layer with Kronecker mixing.
    Dual-branch (x, y) with siamese norms and token-mixing reshape/transpose.
    """
    def __init__(self, seq_len, d_model, block_size, basis_num, d_ff,
                 sinkhorn_iters=20, sinkhorn_temperature=0.2, dropout=0.1):
        super(UniMixerLiteLayer, self).__init__()
        total_dim = seq_len * d_model
        self.seq_len = seq_len
        self.d_model = d_model
        self.total_dim = total_dim

        self.mixer = KroneckerMixer(
            total_dim=total_dim,
            block_size=block_size,
            basis_num=basis_num,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_temperature=sinkhorn_temperature,
            dropout=dropout
        )

        self.ffn = TokenSpecificSwishGLU(seq_len, d_model, d_ff, dropout)

        # Siamese dual-branch norms
        self.siamese_norm_pre_mixing = nn.LayerNorm(d_model)
        self.post_norm_mixing = nn.LayerNorm(d_model)
        self.siamese_norm_pre_pffn = nn.LayerNorm(d_model)
        self.post_norm_moe = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_x, src_y):
        """
        Args:
            src_x: [B, T, D]
            src_y: [B, T, D]
        Returns:
            out_x: [B, T, D]
            out_y: [B, T, D]
        """
        B, T, D = src_x.shape

        # 1. Siamese norm pre mixing
        normed_y_mix = self.siamese_norm_pre_mixing(src_y)
        mixing_input = src_x + normed_y_mix

        # 2. Token mixing: reshape/transpose as in the reference TF code
        # if D % T == 0:

        mixing_h = mixing_input.view(B, T, T, D // T)
        mixing_h = mixing_h.transpose(1, 2)  # [B, T, T, D//T]
        mixing_h = mixing_h.reshape(B, T, D)

        mixing_h_flat = mixing_h.reshape(B, T * D)

        # else:
        #     mixing_h_flat = mixing_input.reshape(B, T * D)

        # 3. Kronecker mixing
        mixing_h_flat = self.mixer(mixing_h_flat)
        mixing_h = mixing_h_flat.view(B, T, D)

        # 4. Post mixing norm + residual
        mid_x = self.post_norm_mixing(src_x + mixing_h)
        mid_y = src_y + mixing_h

        # 5. Siamese norm pre pffn
        normed_y = self.siamese_norm_pre_pffn(mid_y)
        pffn_input = mid_x + normed_y

        # 6. Token-specific SwishGLU
        ffn_out = self.ffn(pffn_input)
        # ffn_out = self.dropout(ffn_out)

        # 7. Post moe norm + residual
        out_x = self.post_norm_moe(mid_x + ffn_out)
        out_y = mid_y + ffn_out

        return out_x, out_y


class UniMixerLite(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="UniMixerLite",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 ffn_in_dim=None,
                 num_transformer_layers=3,
                 block_size=3,
                 basis_num=8,
                 sinkhorn_iters=1,
                 sinkhorn_temperature=0.2,
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
            num_transformer_layers: Number of UniMixer-Lite encoder layers.
            block_size: Chunk size for Kronecker mixing. Defaults to 3.
            basis_num: Number of basis matrices for local interaction W.
            sinkhorn_iters: Number of Sinkhorn iterations.
            sinkhorn_temperature: Temperature for Sinkhorn normalization. Defaults to 0.2.
            transformer_dropout: Dropout rate inside encoder layers and output MLP.
            ffn_out_dim: Hidden dimension of the token-specific SwishGLU.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")
        super(UniMixerLite, self).__init__(
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

        assert (self.seq_len * self.d_model) % block_size == 0, \
            f"seq_len * d_model ({self.seq_len * self.d_model}) must be divisible by block_size ({block_size})"
        self.block_size = block_size

        # Positional Embedding
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        # UniMixer-Lite Encoder Layers
        self.transformer_encoder = nn.ModuleList([
            UniMixerLiteLayer(
                seq_len=self.seq_len,
                d_model=self.d_model,
                block_size=block_size,
                basis_num=basis_num,
                d_ff=ffn_out_dim,
                sinkhorn_iters=sinkhorn_iters,
                sinkhorn_temperature=sinkhorn_temperature,
                dropout=transformer_dropout
            )
            for _ in range(num_transformer_layers)
        ])

        self.output_mlp = nn.Linear(self.d_model, 1)

        # FuxiCTR lifecycle methods
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def set_sinkhorn_temperature(self, temperature):
        """Update the sinkhorn temperature in all KroneckerMixer modules."""
        for layer in self.transformer_encoder:
            layer.mixer.sinkhorn_temperature = temperature

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
        if self.embedding_proj is not None:
            feature_emb = self.embedding_proj(feature_emb)  # [B, L, proj_dim]
        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)  # [B, 2L, d_model]
        x = feature_emb
        y = feature_emb

        # 2. Optional positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding
            y = y + self.pos_embedding

        # 3. UniMixer-Lite Encoder
        for layer in self.transformer_encoder:
            x, y = layer(x, y)

        # 4. Aggregation: use main branch x, mean pooling over tokens
        output = x.mean(dim=1)  # (batch, ffn_in_dim)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
