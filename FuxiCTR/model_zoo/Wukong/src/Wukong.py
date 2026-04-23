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
A Wukong-based CTR prediction model for FuxiCTR.

Architecture:
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection (embedding_dim -> ffn_in_dim)
        -> [Optional double_tokens]
        -> Wukong Encoder (WukongCrossBlock + TokenSpecificSwishGLU + Residual + RMSNorm)
        -> Mean pooling over tokens
        -> Output Linear -> Sigmoid

Each Wukong encoder layer contains:
    - WukongCrossBlock (LCB + FMB + Residual + RMSNorm) as feature interaction
    - TokenSpecificSwishGLU as token-specific nonlinear transformation
    - Residual connection + RMSNorm

Adapted from the TensorFlow 1.x implementation in wukong.py.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.scale * x * torch.rsqrt(ms + self.eps)


class WukongLinearCompressBlock(nn.Module):
    """
    Linear Compress Block (LCB).
    Compresses/expands the number of embedding tokens via a learned weight matrix.
    """
    def __init__(self, num_emb_in, num_emb_out):
        super(WukongLinearCompressBlock, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_emb_in, num_emb_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        """
        Args:
            inputs: [B, num_emb_in, dim_emb]
        Returns:
            outputs: [B, num_emb_out, dim_emb]
        """
        outputs = torch.transpose(inputs, 1, 2)
        outputs = torch.matmul(outputs, self.weight)
        outputs = torch.transpose(outputs, 1, 2)
        return outputs


class WukongFactorizationMachineBlock(nn.Module):
    """
    Factorization Machine Block (FMB).
    Implements low-rank feature interaction via X @ (X^T @ Y), followed by an MLP.
    """
    def __init__(self, token_num, token_dim, rank, ffn_out_dim, num_emb_out, dropout=0.0):
        super(WukongFactorizationMachineBlock, self).__init__()
        self.token_num = token_num
        self.token_dim = token_dim
        self.rank = rank
        self.num_emb_out = num_emb_out

        # Projection matrix Y: [token_num, rank]
        self.weight = nn.Parameter(torch.empty(token_num, rank))
        self.norm = RMSNorm(token_num * rank)

        # MLP: token_num * rank -> ffn_out_dim -> num_emb_out * token_dim
        self.mlp = nn.Sequential(
            nn.Linear(token_num * rank, ffn_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_out_dim, num_emb_out * token_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        """
        Args:
            inputs: [B, token_num, token_dim]
        Returns:
            outputs: [B, num_emb_out, token_dim]
        """
        # Step 1: Compute X^T * Y -> [B, token_dim, rank]
        x_t_y = torch.matmul(torch.transpose(inputs, 1, 2), self.weight)
        # Step 2: Compute X * (X^T * Y) -> [B, token_num, rank]
        outputs = torch.matmul(inputs, x_t_y)
        # Flatten -> [B, token_num * rank]
        outputs = outputs.view(outputs.size(0), -1)
        # RMSNorm
        outputs = self.norm(outputs)
        # MLP
        outputs = self.mlp(outputs)
        # Reshape -> [B, num_emb_out, token_dim]
        outputs = outputs.view(outputs.size(0), self.num_emb_out, self.token_dim)
        return outputs


class WukongCrossBlock(nn.Module):
    """
    Wukong Cross Block: LCB + FMB + concat + residual + RMSNorm.
    Serves as the feature interaction module.
    """
    def __init__(self, num_emb_in, dim_emb, num_emb_lcb, num_emb_fmb, rank_fmb, ffn_out_dim, dropout=0.0):
        super(WukongCrossBlock, self).__init__()
        self.lcb = WukongLinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = WukongFactorizationMachineBlock(
            token_num=num_emb_in,
            token_dim=dim_emb,
            rank=rank_fmb,
            ffn_out_dim=ffn_out_dim,
            num_emb_out=num_emb_fmb,
            dropout=dropout
        )
        self.num_emb_out = num_emb_lcb + num_emb_fmb

        # Residual projection if input/output token counts differ
        if num_emb_in != self.num_emb_out:
            self.residual_proj = nn.Linear(num_emb_in, self.num_emb_out, bias=False)
        else:
            self.residual_proj = None

        self.norm = RMSNorm(dim_emb)

    def forward(self, inputs):
        """
        Args:
            inputs: [B, num_emb_in, dim_emb]
        Returns:
            outputs: [B, num_emb_out, dim_emb]
        """
        lcb_out = self.lcb(inputs)       # [B, num_emb_lcb, dim_emb]
        fmb_out = self.fmb(inputs)       # [B, num_emb_fmb, dim_emb]
        outputs = torch.cat([fmb_out, lcb_out], dim=1)  # [B, num_emb_out, dim_emb]

        # Residual connection
        if self.residual_proj is not None:
            res_out = torch.transpose(inputs, 1, 2)       # [B, dim_emb, num_emb_in]
            res_out = self.residual_proj(res_out)          # [B, dim_emb, num_emb_out]
            res_out = torch.transpose(res_out, 1, 2)       # [B, num_emb_out, dim_emb]
        else:
            res_out = inputs

        outputs = outputs + res_out
        outputs = self.norm(outputs)
        return outputs


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


class WukongLayer(nn.Module):
    """
    Wukong Encoder Layer.
    Combines WukongCrossBlock (feature interaction) and TokenSpecificSwishGLU (nonlinear FFN)
    with residual connections and RMSNorm.
    """
    def __init__(self, num_emb_in, dim_emb, num_emb_lcb, num_emb_fmb, rank_fmb, ffn_out_dim, dropout=0.0):
        super(WukongLayer, self).__init__()
        self.cross = WukongCrossBlock(
            num_emb_in=num_emb_in,
            dim_emb=dim_emb,
            num_emb_lcb=num_emb_lcb,
            num_emb_fmb=num_emb_fmb,
            rank_fmb=rank_fmb,
            ffn_out_dim=ffn_out_dim,
            dropout=dropout
        )
        self.ffn = TokenSpecificSwishGLU(
            seq_len=num_emb_lcb + num_emb_fmb,
            d_model=dim_emb,
            d_ff=ffn_out_dim,
            dropout=dropout
        )

        self.norm2 = RMSNorm(dim_emb)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: [B, num_emb_in, dim_emb]
        Returns:
            out: [B, num_emb_out, dim_emb]
        """
        # Wukong cross block (feature interaction)
        src = self.cross(src)  # [B, num_emb_out, dim_emb]

        # Token-specific SwishGLU FFN + residual + norm
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class WuKong(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="WuKong",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 ffn_in_dim=None,
                 num_transformer_layers=3,
                 num_emb_lcb=16,
                 num_emb_fmb=16,
                 rank_fmb=24,
                 ffn_out_dim=128,
                 transformer_dropout=0.05,
                 output_mlp_hidden_units=None,
                 net_dropout=0.0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 double_tokens=False,
                 **kwargs):
        # Pop legacy HeteroAttention params to avoid passing them to BaseModel
        kwargs.pop("num_heads", None)
        kwargs.pop("att_emb_size", None)
        kwargs.pop("use_cls_token", None)
        kwargs.pop("use_pos_embedding", None)
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")

        super(WuKong, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs)

        self.double_tokens = double_tokens
        if self.double_tokens:
            assert embedding_dim % 2 == 0, f"embedding_dim ({embedding_dim}) must be even for token splitting"
        self.emb_dim = embedding_dim

        # Embedding projection: embedding_dim -> ffn_in_dim
        if ffn_in_dim is not None and ffn_in_dim != self.emb_dim:
            self.embedding_proj = nn.Linear(self.emb_dim, ffn_in_dim)
            proj_dim = ffn_in_dim
        else:
            self.embedding_proj = None
            proj_dim = self.emb_dim

        if self.double_tokens:
            assert proj_dim % 2 == 0, f"proj_dim ({proj_dim}) must be even for token splitting"
        self.d_model = proj_dim // 2 if self.double_tokens else proj_dim
        self.embedding_layer = FeatureEmbedding(feature_map, self.emb_dim)

        self.num_fields = feature_map.num_fields
        self.seq_len = self.num_fields * (2 if self.double_tokens else 1)

        # Wukong Backbone Layers
        self.transformer_encoder = nn.ModuleList()
        for i in range(num_transformer_layers):
            if i == 0:
                num_emb_in = self.seq_len
            else:
                num_emb_in = num_emb_lcb + num_emb_fmb

            self.transformer_encoder.append(
                WukongLayer(
                    num_emb_in=num_emb_in,
                    dim_emb=self.d_model,
                    num_emb_lcb=num_emb_lcb,
                    num_emb_fmb=num_emb_fmb,
                    rank_fmb=rank_fmb,
                    ffn_out_dim=ffn_out_dim,
                    dropout=transformer_dropout
                )
            )

        # Mean pooling over tokens -> output MLP takes d_model
        self.output_mlp = nn.Linear(self.d_model, 1)

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
        if self.embedding_proj is not None:
            feature_emb = self.embedding_proj(feature_emb)
        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)

        x = feature_emb  # [B, seq_len, d_model]

        # 2. Wukong Encoder
        for layer in self.transformer_encoder:
            x = layer(x)  # [B, num_emb_out, d_model]

        # 3. Mean pooling over tokens (original Wukong style)
        output = x.mean(dim=1)  # [B, d_model]

        # 4. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}


# Backward compatibility alias
HeteroAttention = WuKong
