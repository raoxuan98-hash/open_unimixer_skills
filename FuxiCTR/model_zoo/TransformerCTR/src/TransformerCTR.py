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
A Transformer-based CTR prediction model prototype for FuxiCTR.

Architecture:
    FeatureEmbedding (dim=embedding_dim)
        -> Embedding Projection (embedding_dim -> ffn_in_dim)
        -> [Optional CLS Token] + [Optional Positional Embedding]
        -> TransformerEncoder (MultiHeadAttention + FFN + Residual)
        -> [CLS token output OR Flatten all tokens]
        -> MLP -> Sigmoid

You can modify this prototype to implement your own Transformer variants,
e.g., changing the attention mechanism, adding feature-wise masks, or
using different pooling strategies.
"""

import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


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


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with MultiheadAttention and TokenSpecificSwishGLU FFN.
    Post-LayerNorm architecture.
    """
    def __init__(self, seq_len, d_model, num_heads, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = TokenSpecificSwishGLU(seq_len, d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerCTR(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="TransformerCTR",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 ffn_in_dim=None,
                 num_transformer_layers=3,
                 num_heads=2,
                 transformer_dropout=0.1,
                 ffn_out_dim=128,
                 use_cls_token=True,
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
            num_transformer_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            transformer_dropout: Dropout rate inside Transformer layers.
            ffn_out_dim: Hidden dimension of the Feed-Forward Network in Transformer.
            use_cls_token: If True, prepend a learnable CLS token and use its output
                           for final prediction (similar to BERT).
            use_pos_embedding: If True, add learnable positional embeddings.
            output_mlp_hidden_units: Hidden units of the final output MLP.
            net_dropout: Dropout rate of the output MLP.
            batch_norm: Whether to use batch normalization in the output MLP.
        """
        if "ffn_dim" in kwargs:
            ffn_out_dim = kwargs.pop("ffn_dim")
        super(TransformerCTR, self).__init__(
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
        self.use_cls_token = use_cls_token
        self.use_pos_embedding = use_pos_embedding

        self.num_fields = feature_map.num_fields
        self.seq_len = self.num_fields * (2 if self.double_tokens else 1) + (1 if use_cls_token else 0)

        # CLS token: learnable parameter of shape [1, 1, d_model]
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Positional Embedding: learnable parameter of shape [1, seq_len, d_model]
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))

        # Transformer Encoder Layers with TokenSpecificSwishGLU FFN
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                seq_len=self.seq_len,
                d_model=self.d_model,
                num_heads=num_heads,
                dim_feedforward=ffn_out_dim,
                dropout=transformer_dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # Output MLP
        if self.use_cls_token:
            mlp_input_dim = self.d_model
        else:
            mlp_input_dim = self.num_fields * (2 if self.double_tokens else 1) * self.d_model

        self.output_mlp = nn.Linear(mlp_input_dim, 1)

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

        # 1. Feature Embedding: (batch_size, num_fields, 10)
        feature_emb = self.embedding_layer(X)  # [B, L, emb_dim]
        if self.embedding_proj is not None:
            feature_emb = self.embedding_proj(feature_emb)  # [B, L, proj_dim]
        if self.double_tokens:
            B = feature_emb.size(0)
            feature_emb = feature_emb.view(B, -1, self.d_model)  # [B, 2L, d_model]

        # 2. Optional: prepend CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(feature_emb.size(0), -1, -1)
            x = torch.cat([cls_tokens, feature_emb], dim=1)  # (batch, seq_len, d_model)
        else:
            x = feature_emb

        # 3. Optional: add positional embeddings
        if self.use_pos_embedding:
            x = x + self.pos_embedding

        # 4. Transformer Encoder
        # For CTR data, there is typically no padding inside the feature sequence,
        # so we don't need src_key_padding_mask here.
        for layer in self.transformer_encoder:
            x = layer(x)  # (batch, seq_len, d_model)
        transformer_out = x

        # 5. Aggregation strategy
        if self.use_cls_token:
            # Use the output corresponding to the CLS token
            output = transformer_out[:, 0, :]  # (batch, ffn_in_dim)
        else:
            # Flatten all feature-token outputs
            output = transformer_out.flatten(start_dim=1)  # (batch, num_fields * ffn_in_dim)

        # 6. Final prediction
        y_pred = self.output_mlp(output)
        y_pred = self.output_activation(y_pred)

        return {"y_pred": y_pred}
