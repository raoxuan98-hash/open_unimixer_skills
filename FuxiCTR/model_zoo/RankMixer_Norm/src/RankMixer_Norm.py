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
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection (embedding_dim -> ffn_in_dim)
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

from .spherical_ops import NPerTokenSwishGLU, NormalizedResidualUpdate


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


class RankMixerLayer(nn.Module):
    """
    RankMixer encoder layer with spherical normalization:
        1. Rule-based mixing (reshape + transpose)
        2. NormalizedResidualUpdate (replaces LayerNorm + residual)
        3. NPerTokenSwishGLU (L2-normalized per-token SwishGLU)
        4. NormalizedResidualUpdate
    """
    def __init__(self, seq_len, d_model, d_ff, dropout=0.1):
        super(RankMixerLayer, self).__init__()
        assert d_model % seq_len == 0, \
            f"d_model ({d_model}) must be divisible by seq_len ({seq_len}) for rule-based mixing"
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_ffn = NPerTokenSwishGLU(seq_len, d_model, d_ff, dropout)
        self.mixing_residual = NormalizedResidualUpdate(
            num_tokens=seq_len, in_dim=d_model, alpha_init=0.10, gate_type="token_wise"
        )
        self.ffn_residual = NormalizedResidualUpdate(
            num_tokens=seq_len, in_dim=d_model, alpha_init=0.10, gate_type="token_wise"
        )
        
        self.norm = nn.LayerNorm(d_model)

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
        mixed = src + mixing_h
        mixed = F.normalize(mixed, p=2, dim=-1)
        # NPerTokenSwishGLU (output is already L2-normalized)
        ffn_out = self.token_ffn(mixed)
        out = self.ffn_residual(mixed, ffn_out)
        out = self.norm(out)
        return out


class RankMixer(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="RankMixer",
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
                 **kwargs):
        """
        Args:
            feature_map: FeatureMap object from FuxiCTR.
            embedding_dim: Dimension of feature embeddings.
            ffn_in_dim: Dimension projected to before feeding into encoder layers.
                           Must be divisible by num_fields.
            num_transformer_layers: Number of RankMixer layers.
            transformer_dropout: Dropout rate inside RankMixer layers and output MLP.
            ffn_out_dim: Hidden dimension of the token-specific SwishGLU.
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")
        super(RankMixer, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs)

        self.double_tokens = double_tokens
        self.num_fields = feature_map.num_fields
        self.seq_len = self.num_fields * (2 if self.double_tokens else 1)

        # 保存原始 embedding_dim（因为预训练特征可能固定维度，不能随意调整）
        self.raw_embedding_dim = embedding_dim

        # 计算 d_model，使其能被 seq_len 整除
        # 如果指定了 ffn_in_dim，则以 ffn_in_dim 作为投影后的总维度
        if self.double_tokens:
            base_dim = ffn_in_dim if ffn_in_dim is not None else embedding_dim
            target_d_model = base_dim // 2
            if target_d_model % self.seq_len != 0:
                target_d_model = ((target_d_model // self.seq_len) + 1) * self.seq_len
                print(f"[RankMixer] Adjusted d_model to {target_d_model} to be divisible by seq_len ({self.seq_len}) with double_tokens")
            self.d_model = target_d_model
            proj_out_dim = self.d_model * 2
        else:
            base_dim = ffn_in_dim if ffn_in_dim is not None else embedding_dim
            target_d_model = base_dim
            if target_d_model % self.seq_len != 0:
                target_d_model = ((target_d_model // self.seq_len) + 1) * self.seq_len
                print(f"[RankMixer] Adjusted d_model to {target_d_model} to be divisible by seq_len ({self.seq_len})")
            self.d_model = target_d_model
            proj_out_dim = self.d_model

        self.embedding_dim = self.raw_embedding_dim
        self.embedding_layer = FeatureEmbedding(feature_map, self.embedding_dim)
        self.embedding_proj = nn.Linear(self.embedding_dim, proj_out_dim)
        self.use_pos_embedding = use_pos_embedding

        # Positional Embedding: learnable parameter of shape [1, seq_len, embedding_dim]
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        # RankMixer Layers
        self.transformer_encoder = nn.ModuleList([
            RankMixerLayer(
                seq_len=self.seq_len,
                d_model=self.d_model,
                d_ff=ffn_out_dim,
                dropout=transformer_dropout
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
        feature_emb = self.embedding_layer(X)  # [B, L, raw_embedding_dim]
        feature_emb = self.embedding_proj(feature_emb)  # [B, L, proj_out_dim]
        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)  # [B, 2L, d_model]
        
        x = feature_emb
        # 2. Optional: add positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        x = F.normalize(x, dim=-1)
        
        # 3. RankMixer Encoder
        for layer in self.transformer_encoder:
            x = layer(x)  # (batch, seq_len, embedding_dim)

        # x = self.output_norm(x)
        # 4. Aggregation: mean pooling over tokens
        output = x.mean(dim=1)  # (batch, embedding_dim)

        # 5. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
