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
A TokenMixer-Large based CTR prediction model for FuxiCTR.

Architecture:
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection (embedding_dim -> ffn_in_dim)
        -> [Optional Positional Embedding]
        -> TokenMixer-Large Encoder Layers (Dual per-token SwishGLU with
           spherical normalization: L2-normalized weights + learnable residual alpha)
        -> Mean pooling over tokens
        -> Linear -> Sigmoid

This version replaces the original LayerNorm + standard residual with
nGPT-style spherical operators (L2 normalization + NGPTScale + sigmoid-gated
residual updates) ported from unimixer_norm_tf.py.
"""

import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding

from .spherical_ops import (
    NPerTokenNoGLU_Basis,
    NPerTokenSwishGLU,
    NPerTokenSwishGLU_Basis,
    NormalizedResidualUpdate,
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
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        gate = torch.einsum('bld,ldf->blf', x, self.W_gate)
        up = torch.einsum('bld,ldf->blf', x, self.W_up)
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)
        out = torch.einsum('blf,lfd->bld', hidden, self.W_down)
        return out


class UniMixerBlock(nn.Module):
    """
    广义的 UniMixer 块：token 重新划分 + 局部投影 + 基函数加权全局 FFN。

    对应 unimixer_norm_tf.py 中的 generailized_unimixer_block。

    Args:
        total_dim: 总维度，必须能被 block_size 整除。
        block_size: 每个 block 的维度 K。
        token_num_after_reshape: 全局 FFN 的中间维度。
        seq_ffn_basis_num: 局部基矩阵数量。
        token_ffn_basis_num: 全局 FFN 基函数数量。
        dropout: dropout 概率。
    """
    def __init__(self, total_dim, block_size, token_num_after_reshape,
                 seq_ffn_basis_num, token_ffn_basis_num, dropout=0.0):
        super(UniMixerBlock, self).__init__()
        assert total_dim % block_size == 0, \
            f"total_dim ({total_dim}) must be divisible by block_size ({block_size})"
        self.N = total_dim // block_size
        self.K = block_size
        self.token_num_after_reshape = token_num_after_reshape

        # 局部变换: W_1 [N, seq_ffn_basis_num] @ W_V [seq_ffn_basis_num, K, K]
        self.W_1 = nn.Parameter(torch.empty(self.N, seq_ffn_basis_num))
        self.W_V = nn.Parameter(torch.empty(seq_ffn_basis_num, self.K, self.K))


        self.W_global = nn.Parameter(torch.empty(self.K, self.N, self.N))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_global)

    def forward(self, x):
        """
        Args:
            x: [B, total_dim]
        Returns:
            out: [B, total_dim]
        """
        B = x.shape[0]
        N, K = self.N, self.K

        # 局部变换权重
        W_local = torch.einsum('nk,kde->nde', self.W_1, self.W_V)
        nW_local = F.normalize(W_local, p=2, dim=1)

        # Token 重新划分: [B, total_dim] -> [B, N, K]
        x_reshaped = x.view(B, N, K)
        x_reshaped = F.normalize(x_reshaped, p=2, dim=-1)

        # 局部投影
        x_local = torch.einsum('bnk,nkh->bnh', x_reshaped, nW_local)
        x_local = F.normalize(x_local, p=2, dim=-1)

        # 转置以进行全局混合: [B, K, N]
        x_local_trans = x_local.transpose(1, 2)
        x_local_trans = F.normalize(x_local_trans, p=2, dim=-1)

        # 全局投影
        nW_global = F.normalize(self.W_global, p=2, dim=1)
        x_global = torch.einsum('bkn,knm->bkm', x_local_trans, nW_global)   # [B, K, N]

        # 转置回来: [B, N, K]
        x_global = x_global.transpose(1, 2)
        x_global = F.normalize(x_global, p=2, dim=-1)

        return x_global.reshape(B, N * K)


class UniMixerNormLayer(nn.Module):
    """
    TokenMixer-Large Encoder Layer with spherical normalization.

    1. Sequence-mixer: apply NPerTokenSwishGLU on the sequence dimension
       via transpose, i.e. treat feature dim as token positions.
       Weights are L2-normalized; outputs are L2-normalized.

    2. Feature-mixer: NPerTokenSwishGLU on the feature (token) dimension.

    Each sub-layer uses:
        - Pre-L2-normalization (replaces LayerNorm)
        - Spherical SwishGLU mixing
        - NormalizedResidualUpdate (sigmoid-gated alpha + L2 normalize)
    """
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1,
                 use_basis=False, basis_num=8,
                 block_size=None, seq_ffn_basis_num=None,
                 token_num_after_reshape=None):
        super(UniMixerNormLayer, self).__init__()

        total_dim = seq_len * d_model

        if token_num_after_reshape is None:
            token_num_after_reshape = total_dim // block_size
        if seq_ffn_basis_num is None:
            seq_ffn_basis_num = 4

        self.seq_mixer = UniMixerBlock(
            total_dim=total_dim,
            block_size=block_size,
            token_num_after_reshape=token_num_after_reshape,
            seq_ffn_basis_num=seq_ffn_basis_num,
            token_ffn_basis_num=basis_num,
            dropout=dropout,
        )


        # Feature-mixer
        if use_basis:
            self.feature_mixer = NPerTokenSwishGLU_Basis(
                num_tokens=seq_len,
                in_dim=d_model,
                hidden_dim=d_ff,
                basis_num=basis_num,
                dropout=dropout,
            )
        else:
            self.feature_mixer = NPerTokenSwishGLU(
                num_tokens=seq_len,
                in_dim=d_model,
                hidden_dim=d_ff,
                dropout=dropout,
            )

        # Normalized residual updates.
        # gate_type="token_wise" gives alpha shape [seq_len, 1] for the feature-mixer residual,
        # and we reuse the same gate_type for both (each residual operates on [B, seq_len, d_model]).
        self.seq_residual = NormalizedResidualUpdate(
            num_tokens=seq_len,
            in_dim=d_model,
            alpha_init=0.05,
            gate_type="token_wise",
        )
        self.feature_residual = NormalizedResidualUpdate(
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
        # ---- Sub-layer 1: sequence-mixer (spherical pre-norm) ----
        if isinstance(self.seq_mixer, UniMixerBlock):
            # UniMixerBlock: token 重新划分 + 局部变换 + 全局 FFN
            B, T, D = src.shape
            src2 = src.reshape(B, T * D)
            src2 = self.seq_mixer(src2)
            src2 = src2.view(B, T, D)
            src2 = F.normalize(src2, p=2, dim=-1)
        else:
            # 旧的 sequence-mixer: transpose + per-token SwishGLU
            # transpose so that feature dim becomes "token positions" and seq dim becomes "features"
            src2 = src.transpose(1, 2)                # [B, D, T]
            src2 = F.normalize(src2, p=2, dim=-1)          # L2 norm on the new token dim
            src2 = self.seq_mixer(src2)                    # [B, D, T]
            src2 = src2.transpose(1, 2)                    # [B, T, D]
            src2 = F.normalize(src2, p=2, dim=-1)          # L2 norm on the token dim
        src = self.seq_residual(src, src2)             # sigmoid-gated residual + L2 norm
        # ---- Sub-layer 2: feature-mixer (spherical pre-norm) ----
        src2 = self.feature_mixer(src)            # [B, T, D]
        src = self.feature_residual(src, src2)         # sigmoid-gated residual + L2 norm

        return src


class UniMixerNorm(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="TokenMixerLarge",
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
                 block_size=None,
                 seq_ffn_basis_num=None,
                 token_num_after_reshape=None,
                 **kwargs):
        """
        Args:
            feature_map: FeatureMap object from FuxiCTR.
            embedding_dim: Dimension of feature embeddings.
            ffn_in_dim: Dimension projected to before feeding into encoder layers.
            num_transformer_layers: Number of TokenMixer-Large encoder layers.
            transformer_dropout: Dropout rate inside encoder layers and output MLP.
            ffn_out_dim: Hidden dimension of the per-token SwishGLU.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
            double_tokens: If True, split each embedding into two tokens.
            use_basis: If True, use basis-weighted spherical SwishGLU (parameter-efficient).
            basis_num: Number of basis matrices when use_basis=True.
            block_size: Chunk size for token re-partitioning in UniMixerBlock.
                        If provided, enables token re-partitioning and local transformation.
            seq_ffn_basis_num: Number of basis matrices for local interaction W.
            token_num_after_reshape: Hidden dimension of the global FFN inside UniMixerBlock.
        """
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")
        super(UniMixerNorm, self).__init__(
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

        # Positional Embedding
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        # TokenMixer-Large Encoder Layers (spherical version)
        self.transformer_encoder = nn.ModuleList([
            UniMixerNormLayer(
                seq_len=self.seq_len,
                d_model=self.d_model,
                d_ff=ffn_out_dim,
                dropout=transformer_dropout,
                use_basis=use_basis,
                basis_num=basis_num,
                block_size=block_size,
                seq_ffn_basis_num=seq_ffn_basis_num,
                token_num_after_reshape=token_num_after_reshape,
            )
            for _ in range(num_transformer_layers)
        ])
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Linear(self.d_model, 1)
        )

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
        # feature_emb = F.normalize(feature_emb, p=2, dim=-1)  # [B, L, emb_dim]

        if self.embedding_proj is not None:
            feature_emb = self.embedding_proj(feature_emb)  # [B, L, proj_dim]

        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)  # [B, 2L, d_model]
        
        x = feature_emb

        # 2. Optional positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        x = F.normalize(x, p=2, dim=-1)  # [B, 2L, d_model]
        
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

# Backward compatibility alias
TokenMixerLarge = UniMixerNorm
