# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# =========================================================================
"""
LayerNorm-based operators for UniMixerLayerNorm.

Replaces spherical L2 normalization with standard nn.LayerNorm while
keeping the per-token SwishGLU structure and sigmoid-gated residual updates.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGPTScale(nn.Module):
    """
    nGPT-style decoupled scale parameter.
    """
    def __init__(self, shape, scale_init=1.0, scale_alpha=1.0, abs=True):
        super(NGPTScale, self).__init__()
        self.scale_alpha = scale_alpha
        self.abs = abs
        if isinstance(shape, int):
            shape = (shape,)
        self.trainable_scale = nn.Parameter(torch.full(shape, scale_init, dtype=torch.float32))

    def forward(self):
        if self.abs:
            return torch.abs(self.trainable_scale) * self.scale_alpha
        else:
            return self.trainable_scale * self.scale_alpha


def reverse_sigmoid(x):
    """Inverse sigmoid for initializing alpha meta-parameter."""
    return np.log(x / (1.0 - x))


class LayerNormResidualUpdate(nn.Module):
    """
    Sigmoid-gated weighted residual update with LayerNorm.

    h <- LayerNorm( (1 - alpha) * h + alpha * block_output )

    Supports gate types:
        global:       alpha shape [1]
        token_wise:   alpha shape [num_tokens, 1]
        feature_wise: alpha shape [in_dim]
        full:         alpha shape [num_tokens, in_dim]
    """
    def __init__(self, num_tokens, in_dim, alpha_init=0.05, gate_type="token_wise"):
        super(LayerNormResidualUpdate, self).__init__()
        self.alpha_init = alpha_init
        self.gate_type = gate_type

        meta_alpha = reverse_sigmoid(alpha_init)

        if gate_type == "global":
            self.alpha = NGPTScale((1,), scale_init=0.05,
                                   scale_alpha=meta_alpha / 0.05, abs=True)
        elif gate_type == "token_wise":
            self.alpha = NGPTScale((num_tokens, 1), scale_init=0.05,
                                   scale_alpha=meta_alpha / 0.05, abs=True)
        elif gate_type == "feature_wise":
            self.alpha = NGPTScale((in_dim,), scale_init=0.05,
                                   scale_alpha=meta_alpha / 0.05, abs=True)
        elif gate_type == "full":
            self.alpha = NGPTScale((num_tokens, in_dim), scale_init=0.05,
                                   scale_alpha=meta_alpha / 0.05, abs=True)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}, "
                             f"must be 'global', 'token_wise', 'feature_wise' or 'full'")

        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, hidden_states, block_output):
        """
        Args:
            hidden_states: [B, num_tokens, in_dim]
            block_output:  [B, num_tokens, in_dim]
        Returns:
            out: [B, num_tokens, in_dim]
        """
        alpha = torch.sigmoid(self.alpha())
        out = (1.0 - alpha) * hidden_states + alpha * block_output
        out = self.layer_norm(out)
        return out


class PerTokenSwishGLU(nn.Module):
    """
    Standard per-token SwishGLU (via einsum) without L2 normalization.

    Each token position has its own projection matrices.
    """
    def __init__(self, num_tokens, in_dim, hidden_dim, dropout=0.0):
        super(PerTokenSwishGLU, self).__init__()
        self.num_tokens = num_tokens
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.W_gate = nn.Parameter(torch.empty(num_tokens, in_dim, hidden_dim))
        self.W_up = nn.Parameter(torch.empty(num_tokens, in_dim, hidden_dim))
        self.W_down = nn.Parameter(torch.empty(num_tokens, hidden_dim, in_dim))

        self.u_scale = NGPTScale((1, hidden_dim), scale_init=1.0, scale_alpha=1.0)
        self.v_scale = NGPTScale((1, hidden_dim), scale_init=1.0,
                                 scale_alpha=math.sqrt(in_dim))

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
        u = torch.einsum('bld,ldf->blf', x, self.W_gate)
        v = torch.einsum('bld,ldf->blf', x, self.W_up)

        scaled_u = self.u_scale() * u
        scaled_v = self.v_scale() * v

        h = F.silu(scaled_v) * scaled_u
        h = self.dropout(h)

        out = torch.einsum('blf,lfd->bld', h, self.W_down)
        return out


class PerTokenSwishGLU_Basis(nn.Module):
    """
    Basis-weighted per-token SwishGLU without L2 normalization.
    """
    def __init__(self, num_tokens, in_dim, hidden_dim, basis_num, dropout=0.0):
        super(PerTokenSwishGLU_Basis, self).__init__()
        self.num_tokens = num_tokens
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.basis_num = basis_num

        self.W_gate_basis = nn.Parameter(torch.empty(basis_num, in_dim, hidden_dim))
        self.W_up_basis = nn.Parameter(torch.empty(basis_num, in_dim, hidden_dim))
        self.W_down_basis = nn.Parameter(torch.empty(basis_num, hidden_dim, in_dim))

        self.token_coef1 = nn.Parameter(torch.empty(num_tokens, basis_num))
        self.token_coef2 = nn.Parameter(torch.empty(num_tokens, basis_num))
        self.token_coef3 = nn.Parameter(torch.empty(num_tokens, basis_num))

        self.u_scale = NGPTScale((1, hidden_dim), scale_init=1.0, scale_alpha=1.0)
        self.v_scale = NGPTScale((1, hidden_dim), scale_init=1.0,
                                 scale_alpha=math.sqrt(in_dim))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_gate_basis)
        nn.init.xavier_uniform_(self.W_up_basis)
        nn.init.xavier_uniform_(self.W_down_basis)
        nn.init.xavier_uniform_(self.token_coef1)
        nn.init.xavier_uniform_(self.token_coef2)
        nn.init.xavier_uniform_(self.token_coef3)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            out: [B, L, D]
        """
        coef1 = F.softmax(self.token_coef1, dim=-1)
        coef2 = F.softmax(self.token_coef2, dim=-1)
        coef3 = F.softmax(self.token_coef3, dim=-1)

        W_gate = torch.einsum('lb,bio->lio', coef1, self.W_gate_basis)
        W_up = torch.einsum('lb,bio->lio', coef2, self.W_up_basis)
        W_down = torch.einsum('lb,boj->loj', coef3, self.W_down_basis)

        u = torch.einsum('bld,ldf->blf', x, W_gate)
        v = torch.einsum('bld,ldf->blf', x, W_up)

        scaled_u = self.u_scale() * u
        scaled_v = self.v_scale() * v

        h = F.silu(scaled_v) * scaled_u
        h = self.dropout(h)

        out = torch.einsum('blf,lfd->bld', h, W_down)
        return out
