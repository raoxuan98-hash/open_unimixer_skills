# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# =========================================================================
"""
Spherical operators for nGPT-style normalized neural networks.

Ported from unimixer_norm_tf.py to PyTorch.
Includes:
    - NGPTScale: decoupled parameterization scale (nGPT)
    - NormalizedResidualUpdate: L2-normalized residual with sigmoid-gated alpha
    - NPerTokenSwishGLU: per-token SwishGLU with L2-normalized weights
    - NPerTokenSwishGLU_Basis: basis-weighted variant for parameter efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGPTScale(nn.Module):
    """
    nGPT 推荐的分离参数化缩放系数。

    可训练参数以 scale_init 初始化，实际生效值为:
        scale = trainable_param * scale_alpha  (或 abs(trainable_param) * scale_alpha)

    这使得 Adam 优化器在适中的参数尺度上进行更新，同时通过 scale_alpha
    放大/控制实际作用幅度，解决 Adam 对小尺度参数更新过慢的问题。
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
    """Sigmoid 的反函数，用于从目标 alpha_init 反推元参数初始值。"""
    import numpy as np
    return np.log(x / (1.0 - x))


class NormalizedResidualUpdate(nn.Module):
    """
    nGPT 风格的归一化残差更新。

    h <- Norm( (1-alpha) * h + alpha * block_output )

    与 TF 版本一致：用 reverse_sigmoid(alpha_init) 作为元参数初始值，
    前向时直接 sigmoid(alpha)，保证初始 alpha 精确等于 alpha_init。

    支持四种门控类型:
        global:       全局共享一个 alpha，形状 [1]
        token_wise:   每个 token 共享，alpha 形状 [num_tokens, 1]
        feature_wise: 每个特征维度共享，alpha 形状 [in_dim]
        full:         每个 token 每个特征独立，alpha 形状 [num_tokens, in_dim]
    """
    def __init__(self, num_tokens, in_dim, alpha_init=0.05, gate_type="token_wise"):
        super(NormalizedResidualUpdate, self).__init__()
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

    def forward(self, hidden_states, block_output):
        """
        Args:
            hidden_states: [B, num_tokens, in_dim]
            block_output:  [B, num_tokens, in_dim], 已经归一化
        Returns:
            out: [B, num_tokens, in_dim], L2 归一化后的结果
        """
        alpha = torch.sigmoid(self.alpha())
        # alpha 会自动广播到 [B, num_tokens, in_dim]
        out = (1.0 - alpha) * hidden_states + alpha * block_output
        out = F.normalize(out, p=2, dim=-1)
        return out


class NPerTokenSwishGLU(nn.Module):
    """
    归一化的 per-token SwishGLU (parallel via einsum)。

    每个 token 位置拥有独立的投影矩阵，权重沿嵌入维度(输入轴)做 L2 归一化，
    并通过 ngpt_scale 进行缩放。对应 unimixer_norm_tf.py 中的
    normalized_ffn_swish_glu_per_token。

    Args:
        num_tokens: token 位置数量 L
        in_dim:     每个 token 的输入维度 D
        hidden_dim: 中间层维度 D_ff
        dropout:    dropout 概率
    """
    def __init__(self, num_tokens, in_dim, hidden_dim, dropout=0.0):
        super(NPerTokenSwishGLU, self).__init__()
        self.num_tokens = num_tokens
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # 每个 token 独立的权重: [L, D, D_ff] 或 [L, D_ff, D]
        self.W_gate = nn.Parameter(torch.empty(num_tokens, in_dim, hidden_dim))
        self.W_up = nn.Parameter(torch.empty(num_tokens, in_dim, hidden_dim))
        self.W_down = nn.Parameter(torch.empty(num_tokens, hidden_dim, in_dim))

        # 每个 token 独立的 scale: [L, D_ff]
        # self.u_scale = NGPTScale((num_tokens, hidden_dim), scale_init=1.0, scale_alpha=1.0)
        # self.v_scale = NGPTScale((num_tokens, hidden_dim), scale_init=1.0,
        #                          scale_alpha=math.sqrt(in_dim))

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
            out: [B, L, D], 已沿最后一维 L2 归一化
        """
        # 沿嵌入维度(输入轴, axis=1)对每个 token 的权重归一化
        W_gate_norm = F.normalize(self.W_gate, p=2, dim=1)
        W_up_norm = F.normalize(self.W_up, p=2, dim=1)
        W_down_norm = F.normalize(self.W_down, p=2, dim=1)

        # [B, L, D] @ [L, D, D_ff] -> [B, L, D_ff]
        u = torch.einsum('bld,ldf->blf', x, W_gate_norm)
        v = torch.einsum('bld,ldf->blf', x, W_up_norm)

        scaled_u = self.u_scale() * u
        scaled_v = self.v_scale() * v

        # SwiGLU: h = swish(v) * u
        h = F.silu(scaled_v) * scaled_u
        h = self.dropout(h)

        # [B, L, D_ff] @ [L, D_ff, D] -> [B, L, D]
        out = torch.einsum('blf,lfd->bld', h, W_down_norm)
        out = F.normalize(out, p=2, dim=-1)
        return out


class NPerTokenSwishGLU_Basis(nn.Module):
    """
    基于基函数加权组合的归一化 SwiGLU FFN。

    结合 SwiGLU 门控结构与基函数加权机制，每个 token 通过独立的组合系数
    从共享的基函数矩阵中构建个性化权重，实现细粒度自适应变换。
    对应 unimixer_norm_tf.py 中的 normalized_ffn_swish_glu_basis_weighted。

    Args:
        num_tokens: token 位置数量 L
        in_dim:     每个 token 的输入维度 D
        hidden_dim: 中间层维度 D_ff
        basis_num:  基础权重矩阵的数量
        dropout:    dropout 概率
    """
    def __init__(self, num_tokens, in_dim, hidden_dim, basis_num, dropout=0.0):
        super(NPerTokenSwishGLU_Basis, self).__init__()
        self.num_tokens = num_tokens
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.basis_num = basis_num

        # 基础权重矩阵 (三组，对应 SwiGLU 的 gate、up 和 down)
        self.W_gate_basis = nn.Parameter(torch.empty(basis_num, in_dim, hidden_dim))
        self.W_up_basis = nn.Parameter(torch.empty(basis_num, in_dim, hidden_dim))
        self.W_down_basis = nn.Parameter(torch.empty(basis_num, hidden_dim, in_dim))

        # 每个 token 的组合系数 [L, basis_num]
        self.token_coef1 = nn.Parameter(torch.empty(num_tokens, basis_num))
        self.token_coef2 = nn.Parameter(torch.empty(num_tokens, basis_num))
        self.token_coef3 = nn.Parameter(torch.empty(num_tokens, basis_num))

        # 每个 token 独立的 scale: [L, D_ff]
        # self.u_scale = NGPTScale((num_tokens, hidden_dim), scale_init=1.0, scale_alpha=1.0)
        # self.v_scale = NGPTScale((num_tokens, hidden_dim), scale_init=1.0,
        #                          scale_alpha=math.sqrt(in_dim))

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
            out: [B, L, D], 已沿最后一维 L2 归一化
        """
        # Softmax 归一化组合系数
        coef1 = F.softmax(self.token_coef1, dim=-1)  # [L, basis_num]
        coef2 = F.softmax(self.token_coef2, dim=-1)
        coef3 = F.softmax(self.token_coef3, dim=-1)

        # 加权组合得到每个 token 的个性化权重
        W_gate = torch.einsum('lb,bio->lio', coef1, self.W_gate_basis)   # [L, D, D_ff]
        W_up = torch.einsum('lb,bio->lio', coef2, self.W_up_basis)       # [L, D, D_ff]
        W_down = torch.einsum('lb,boj->loj', coef3, self.W_down_basis)   # [L, D_ff, D]

        # 沿嵌入维度归一化
        W_gate_norm = F.normalize(W_gate, p=2, dim=1)
        W_up_norm = F.normalize(W_up, p=2, dim=1)
        W_down_norm = F.normalize(W_down, p=2, dim=1)

        # SwiGLU 计算
        u = torch.einsum('bld,ldf->blf', x, W_gate_norm)
        v = torch.einsum('bld,ldf->blf', x, W_up_norm)

        scaled_u = self.u_scale() * u
        scaled_v = self.v_scale() * v

        h = F.silu(scaled_v) * scaled_u
        h = self.dropout(h)

        out = torch.einsum('blf,lfd->bld', h, W_down_norm)
        out = F.normalize(out, p=2, dim=-1)
        return out
