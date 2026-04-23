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
HybridMixer: A fusion of RankMixer and UniMixerLite.

Architecture:
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection
        -> [Optional Positional Embedding]
        -> Siamese dual-branch (x and y)
        -> HybridMixer Layers (
               Rule-based mixing (RankMixer) 
               + Kronecker mixing (UniMixerLite) via learnable gate
               + Token-specific SwishGLU
           )
        -> Mean pooling over tokens from main branch x
        -> SwishGLU MLP -> Sigmoid

The learnable fusion gate initializes Kronecker contribution to ~0.05
and lets it optimize via gradients.
"""

import math
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
        # self.register_buffer('shuffle_idx', torch.randperm(total_dim))
        
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
        x_reshaped = x.view(B, N, K)
        x_reshaped = self.norm1(x_reshaped)

        # mixing_h分支：将向量拍平后按固定随机索引打乱，作为残差
        # mixing_h = x.view(B, -1)                    # [B, total_dim]
        # mixing_h = mixing_h[:, self.shuffle_idx]    # [B, total_dim] 随机打乱
        # mixing_h = mixing_h.view(B, N, K)           # 恢复形状

        A_logits_normalized = F.normalize(self.A_logits, p=2, dim=0)
        # A = log_sinkhorn(A_logits_normalized.unsqueeze(0), n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temperature).squeeze(0)
        A = A_logits_normalized
        W_logits = torch.einsum('tk,kde->tde', self.W_1, self.W_V)
        W_logits_normalized = F.normalize(W_logits, p=2, dim=1)
        # W = log_sinkhorn(W_logits_normalized.unsqueeze(0), n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temperature).squeeze(0)
        W = W_logits_normalized
        x_local = torch.einsum('bni,nio->bno', x_reshaped, W)
        x_local = self.norm2(x_local.transpose(1, 2))
        x_global = x_local @ A
        output = x_global.reshape(B, N * K)
        return output


class HybridMixerLayer(nn.Module):
    """
    HybridMixer encoder layer:
        1. Rule-based mixing (RankMixer) on x
        2. Kronecker mixing (UniMixerLite) on x + normed_y
        3. Learnable gate fuses both, init Kronecker contrib ~0.05
        4. Residual + LayerNorm
        5. Token-specific SwishGLU
        6. Residual + LayerNorm
        7. Update y branch
    """
    def __init__(self, seq_len, d_model, d_ff, block_size, basis_num,
                 sinkhorn_iters=20, sinkhorn_temperature=0.2, dropout=0.1):
        super(HybridMixerLayer, self).__init__()
        assert d_model % seq_len == 0, \
            f"d_model ({d_model}) must be divisible by seq_len ({seq_len}) for rule-based mixing"

        self.seq_len = seq_len
        self.d_model = d_model

        # UniMixerLite Kronecker mixer
        total_dim = seq_len * d_model
        self.kronecker_mixer = KroneckerMixer(
            total_dim=total_dim,
            block_size=block_size,
            basis_num=basis_num,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_temperature=sinkhorn_temperature,
            dropout=dropout
        )

        # Token-specific SwishGLU (shared after fusion)
        self.token_ffn = TokenSpecificSwishGLU(seq_len, d_model, d_ff, dropout)

        # Learnable fusion gate.
        # sigmoid(-2.944) ≈ 0.05, so Kronecker starts with small contribution.
        self.mix_logit = nn.Parameter(torch.tensor(-2.944))

        self.norm_y = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        Args:
            x: [B, L, D]  main branch
            y: [B, L, D]  auxiliary branch
        Returns:
            final_x: [B, L, D]
            final_y: [B, L, D]
        """
        B, L, D = x.shape

        # 1. RankMixer rule-based mixing
        rule_h = x.view(B, L, L, D // L)
        rule_h = rule_h.transpose(1, 2).contiguous()
        rule_h = rule_h.view(B, L, D)

        # 2. UniMixerLite Kronecker mixing (siamese input)
        normed_y = self.norm_y(y)
        kronecker_input = x + normed_y
        kronecker_h = self.kronecker_mixer(kronecker_input.reshape(B, L * D))
        kronecker_h = kronecker_h.view(B, L, D)

        # 3. Learnable fusion: alpha init ~0.05 for Kronecker
        # alpha = torch.sigmoid(self.mix_logit)
        mixing_h = 0.5 * rule_h + 0.5 * kronecker_h

        mixed = self.norm1(x + self.dropout(mixing_h))

        # 4. Token-specific SwishGLU
        ffn_out = self.token_ffn(mixed)
        out = self.norm2(mixed + self.dropout(ffn_out))

        # 5. Update y branch
        y_out = y + self.dropout(ffn_out)

        return out, y_out


class HybridMixer(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="HybridMixer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 ffn_in_dim=None,
                 num_transformer_layers=3,
                 block_size=3,
                 basis_num=8,
                 sinkhorn_iters=20,
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
            embedding_dim: Dimension of feature embeddings.
            ffn_in_dim: Dimension projected to before feeding into encoder layers.
            num_transformer_layers: Number of HybridMixer layers.
            block_size: Chunk size for Kronecker mixing.
            basis_num: Number of basis matrices for local interaction W.
            sinkhorn_iters: Number of Sinkhorn iterations.
            sinkhorn_temperature: Temperature for Sinkhorn normalization.
            transformer_dropout: Dropout rate inside encoder layers and output MLP.
            ffn_out_dim: Hidden dimension of the token-specific SwishGLU.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
            double_tokens: If True, split each embedding into two tokens.
        """
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")
        super(HybridMixer, self).__init__(
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

        # Embedding projection
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
        target_d_model = proj_dim // 2 if self.double_tokens else proj_dim

        # d_model must satisfy:
        #   1) d_model % seq_len == 0      (for RankMixer rule-based mixing)
        #   2) (seq_len * d_model) % block_size == 0  (for Kronecker mixing)
        self.d_model = target_d_model
        max_search = target_d_model + 10000
        while self.d_model < max_search:
            if self.d_model % self.seq_len == 0 and (self.seq_len * self.d_model) % block_size == 0:
                break
            self.d_model += 1
        else:
            raise ValueError(
                f"Cannot find d_model >= {target_d_model} satisfying "
                f"d_model % seq_len ({self.seq_len}) == 0 and "
                f"(seq_len * d_model) % block_size ({block_size}) == 0"
            )

        if self.d_model != target_d_model:
            print(f"[HybridMixer] Adjusted d_model from {target_d_model} to {self.d_model} "
                  f"to satisfy both RankMixer and Kronecker constraints")

        if self.double_tokens:
            proj_out_dim = self.d_model * 2
        else:
            proj_out_dim = self.d_model

        # Recreate projection if target dim changed
        if self.embedding_proj is not None:
            if self.embedding_proj.out_features != proj_out_dim:
                self.embedding_proj = nn.Linear(self.emb_dim, proj_out_dim)
        elif proj_out_dim != self.emb_dim:
            self.embedding_proj = nn.Linear(self.emb_dim, proj_out_dim)
        else:
            self.embedding_proj = None

        assert (self.seq_len * self.d_model) % block_size == 0, \
            f"seq_len * d_model ({self.seq_len * self.d_model}) must be divisible by block_size ({block_size})"
        self.block_size = block_size

        # Siamese y-branch projection
        self.y_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Positional Embedding
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        # HybridMixer Encoder Layers
        self.transformer_encoder = nn.ModuleList([
            HybridMixerLayer(
                seq_len=self.seq_len,
                d_model=self.d_model,
                d_ff=ffn_out_dim,
                block_size=block_size,
                basis_num=basis_num,
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
            layer.kronecker_mixer.sinkhorn_temperature = temperature

    def forward(self, inputs):
        X = self.get_inputs(inputs)

        # 1. Feature Embedding
        feature_emb = self.embedding_layer(X)  # [B, L, emb_dim]
        if self.embedding_proj is not None:
            feature_emb = self.embedding_proj(feature_emb)  # [B, L, proj_out_dim]
        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)  # [B, 2L, d_model]
        x = feature_emb
        y = self.y_proj(feature_emb)

        # 2. Optional positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding
            y = y + self.pos_embedding

        # 3. HybridMixer Encoder
        for layer in self.transformer_encoder:
            x, y = layer(x, y)

        # 4. Aggregation: use main branch x, mean pooling over tokens
        output = x.mean(dim=1)  # (batch, d_model)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
