from __future__ import print_function
import numpy as np
import math
import os
import argparse

UDP_OUTPUT_PRED = int(os.getenv('OUTPUT_PRED', 0))
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
args = parser.parse_known_args()[0]

if args.mode == 'train':
    import tensorflow.compat.v1 as tf
    import kai.tensorflow as config
    
    default_param_attr = config.nn.ParamAttr(
        initializer=config.nn.UniformInitializer(0.0001),
        access_method=config.nn.ProbabilityAccess(100.0),
        recycle_method=config.nn.UnseendaysRecycle(delete_after_unseen_days=30, delete_threshold=0.1, allow_dynamic_delete=True)
    )
    config.nn.set_default_param_attr(default_param_attr)
else:
    import tensorflow as tf
    from mio_tensorflow.config import MioConfig
    import mio_tensorflow.patch as mio_tensorflow_patch
    mio_tensorflow_patch.apply()
    base_config = os.path.join(os.path.dirname( os.path.realpath(__file__)), 'base.yaml')
    config = MioConfig.from_base_yaml(base_config, clear_embeddings=True, clear_params=True,
                                  dryrun=False, label_with_kv=True, grad_no_scale=False,
                                  with_kai=False)

##################
def rms_norm(x, eps=1e-8, p=-1., bias=False, scope=None):
    with tf.variable_scope(scope or "rms_norm"):
        layer_size = x.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        if bias:
            offset = tf.get_variable("offset", [layer_size], initializer=tf.zeros_initializer())
        else:
            offset = 0.

        if p < 0. or p > 1.:
            ms = tf.reduce_mean(x ** 2, -1, keep_dims=True)
        else:
            partial_size = int(layer_size * p)
            partial_x, _ = tf.split(x, [partial_size, layer_size - partial_size], axis=-1)

            ms = tf.reduce_mean(partial_x ** 2, -1, keep_dims=True)

        return scale * x * tf.rsqrt(ms + eps) + offset
 
def swish(x):
    return x * tf.nn.sigmoid(x)

def reverse_sigmoid(x):
    return np.log(x / (1 - x))

def normalized_residual_update(hidden_states, block_output, scope, alpha_init=0.05, gate_type="token_wise"):
    """
    nGPT 风格的归一化残差更新。

    h <- Norm( h + alpha * (block_output - h) )

    支持三种门控类型:
    1. feature_wise: 每个特征维度有独立的门控值，alpha 形状为 [d_model]
    2. token_wise: 每个 token 有独立的门控值，跨 feature 共享，alpha 形状为 [l, 1]
    3. full: 每个 token 的每个特征都有独立的门控值，alpha 形状为 [l, d_model]

    Args:
        hidden_states: [B, l, d_model]。
        block_output: 与 hidden_states 同 shape，Block 的输出（需已归一化）。
        scope: 变量名前缀，用于创建 alpha 参数。
        alpha_init: alpha 的初始实际值，默认 0.05（对应论文推荐值，约 1/n_layers）。
        gate_type: "feature_wise", "token_wise" 或 "full"，默认为 "token_wise"。

    Returns:
        out: 与输入同 shape，已沿最后一维 L2 归一化。
    """
    shape_list = hidden_states.get_shape().as_list()
    d_model = shape_list[-1]

    meta_alpha = reverse_sigmoid(alpha_init)
    
    if gate_type == "global":
        alpha = ngpt_scale(
            scope + "_alpha", [1],
            scale_init=meta_alpha,
            scale_alpha=1.0,
            abs=False
        )

    elif gate_type == "token_wise":
        # token-wise 门控: 每个 token 有一个门控值，跨 feature 共享
        l = shape_list[1]  # token 数量
        alpha = ngpt_scale(
            scope + "_alpha", [l, 1],
            scale_init=meta_alpha,
            scale_alpha=1.0,
            abs=False
        )
        # alpha 形状为 [l, 1]，广播后与 [B, l, d_model] 兼容
        
    elif gate_type == "feature_wise":
        # feature-wise 门控: 每个特征维度有一个门控值
        alpha = ngpt_scale(
            scope + "_alpha", [d_model],
            scale_init=meta_alpha,
            scale_alpha=1.0,
            abs=False
        )
        # alpha 形状为 [d_model]，广播后与 [B, l, d_model] 兼容

    elif gate_type == "full":
        # 全量门控: 每个 token 的每个特征都有独立的门控值
        l = shape_list[1]  # token 数量
        alpha = ngpt_scale(
            scope + "_alpha", [l, d_model],
            scale_init=meta_alpha,
            scale_alpha=1.0,
            abs=False
        )
        # alpha 形状为 [l, d_model]，广播后与 [B, l, d_model] 兼容
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}, must be 'feature_wise', 'token_wise' or 'full'")
    
    alpha = tf.nn.sigmoid(alpha)
    if gate_type == "global":
        tf.summary.scalar(f"residual_update_alpha/{scope}", tf.reduce_mean(alpha))
    else:
        tf.summary.scalar(f"residual_update_alpha/{scope}", tf.reduce_mean(alpha))
        tf.summary.scalar(f"residual_update_alpha_std/{scope}", tf.math.reduce_std(alpha))
        tf.summary.scalar(f"residual_update_alpha_max/{scope}", tf.reduce_max(alpha))
        tf.summary.scalar(f"residual_update_alpha_min/{scope}", tf.reduce_min(alpha))

    out = (1.0 - alpha) * hidden_states + alpha * block_output
    out = tf.nn.l2_normalize(out, axis=-1)
    return out

def ngpt_scale(name, shape, scale_init, scale_alpha, abs=True):
    """
    nGPT 推荐的分离参数化缩放系数。

    可训练参数以 scale_init 初始化，实际生效值为:
        scale = trainable_param * scale_alpha

    这使得 Adam 优化器在适中的参数尺度上进行更新，同时通过 scale_alpha 放大/控制
    实际作用幅度，解决 Adam 对小尺度参数更新过慢的问题。

    Args:
        name: 变量名。
        shape: 参数 shape。
        scale_init: 可训练参数的初始值（ Adam 直接优化这个变量）。
        scale_alpha: 固定的常量放大系数。

    Returns:
        Tensor: 实际生效的缩放值。
    """
    trainable_scale = tf.get_variable(
        name, shape, initializer=tf.constant_initializer(scale_init)
    )

    if abs:
        return tf.abs(trainable_scale) * scale_alpha
    else:
        return trainable_scale * scale_alpha


def swish(x):
    """SiLU / Swish 激活函数。"""
    return x * tf.nn.sigmoid(x)

def normalized_ffn_swish_glu(hidden_states,
                             d_model,
                             d_ff,
                             scope="normalized_ffn_swish_glu",
                             reuse=None):
    """
    归一化 SwiGLU FFN (batch-level / token-independent 版本)。

    对应论文 Section 2.4.2。所有权重矩阵沿输入维度(axis=0)归一化，
    u/v 的中间激活不做额外的 L2 归一化，并通过 ngpt_scale 进行缩放。

    Args:
        hidden_states: [B, d_model] 或兼容的 2-D 输入。
        d_model: 输入/输出维度。
        d_ff: 中间层维度。
        scope: variable_scope。
        reuse: 是否复用变量。

    Returns:
        out: 与 hidden_states 同 rank 的输出，最后一维为 d_model，且已 L2 归一化。
    """
    with tf.variable_scope(scope, reuse=reuse):
        w1 = tf.get_variable("w1", [d_model, d_ff],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        w2 = tf.get_variable("w2", [d_model, d_ff],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        w3 = tf.get_variable("w3", [d_ff, d_model],
                             initializer=tf.random_normal_initializer(stddev=0.01))

        # scale params: scale = scale_init * scale_alpha
        u_scale = ngpt_scale("u_scale", [d_ff], scale_init=1.0, scale_alpha=1.0)
        v_scale = ngpt_scale("v_scale", [d_ff], scale_init=1.0,
                             scale_alpha=np.sqrt(d_model))

        # 沿嵌入维度(输入轴, axis=0)归一化权重
        nw1 = tf.nn.l2_normalize(w1, axis=0)
        nw2 = tf.nn.l2_normalize(w2, axis=0)
        nw3 = tf.nn.l2_normalize(w3, axis=0)

        u = tf.matmul(hidden_states, nw1)
        v = tf.matmul(hidden_states, nw2)

        scaled_u = u_scale * u
        scaled_v = v_scale * v

        # SwiGLU
        h = swish(scaled_v) * scaled_u
        out = tf.matmul(h, nw3)
        out = tf.nn.l2_normalize(out, axis=-1)
        return out


def normalized_ffn_swish_glu_per_token(hidden_states, d_model, d_ff,
                                        scope="normalized_ffn_swish_glu_per_token",
                                        reuse=None):
    """
    支持 [B, L, d_model] 输入的并行 SwiGLU FFN，每个 token 拥有独立的权重。

    使用 tf.einsum 一次性对所有 token 进行计算，无需 per-token 循环。
    遵循论文 Section 2.4.2，不对 u/v 做额外的 L2 归一化。
    """
    with tf.variable_scope(scope, reuse=reuse):
        l = hidden_states.get_shape().as_list()[1]

        # 每个 token 独立的权重: [l, d_model, d_ff]
        w1 = tf.get_variable("w1", [l, d_model, d_ff],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        w2 = tf.get_variable("w2", [l, d_model, d_ff],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        w3 = tf.get_variable("w3", [l, d_ff, d_model],
                             initializer=tf.random_normal_initializer(stddev=0.01))

        # 每个 token 独立的 scale: [l, d_ff]
        u_scale = ngpt_scale("u_scale", [l, d_ff], scale_init=1.0, scale_alpha=1.0)
        v_scale = ngpt_scale("v_scale", [l, d_ff], scale_init=1.0,
                             scale_alpha=np.sqrt(d_model))

        tf.summary.scalar("gate_scale", tf.reduce_mean(u_scale))
        tf.summary.scalar("gate_scale_std", tf.math.reduce_std(u_scale))
        tf.summary.scalar("value_scale", tf.reduce_mean(v_scale))
        tf.summary.scalar("value_scale_std", tf.math.reduce_std(v_scale))

        # 沿嵌入维度(输入轴, axis=1)对每个 token 的权重归一化
        nw1 = tf.nn.l2_normalize(w1, axis=1)
        nw2 = tf.nn.l2_normalize(w2, axis=1)
        nw3 = tf.nn.l2_normalize(w3, axis=1)

        # [B, l, d_model] @ [l, d_model, d_ff] -> [B, l, d_ff]
        u = tf.einsum('bli,lio->blo', hidden_states, nw1)
        v = tf.einsum('bli,lio->blo', hidden_states, nw2)

        scaled_u = u_scale * u
        scaled_v = v_scale * v

        # SwiGLU
        h = swish(scaled_v) * scaled_u

        # [B, l, d_ff] @ [l, d_ff, d_model] -> [B, l, d_model]
        out = tf.einsum('blo,loj->blj', h, nw3)
        out = tf.nn.l2_normalize(out, axis=-1)
        return out


def normalized_ffn_no_glu_basis_weighted(hidden_states, d_model, d_ff, token_ffn_basis_num,
                                         scope="normalized_ffn_no_glu_basis_weighted",
                                         reuse=None):
    """
    基于基函数加权组合的无门控归一化 FFN。

    每个 token 拥有独立的基函数组合系数，实现细粒度自适应变换。

    Args:
        hidden_states: [B, l, d_model]，l 为 token 数量。
        d_model: 输入/输出维度。
        d_ff: FFN 中间层维度。
        token_ffn_basis_num: 基础权重矩阵的数量。
        scope: variable_scope 名称。
        reuse: 是否复用变量。

    Returns:
        out: [B, l, d_model]，已 L2 归一化。
    """
    with tf.variable_scope(scope, reuse=reuse):
        l = hidden_states.get_shape().as_list()[1]

        # 基础权重矩阵
        w1_basis = tf.get_variable("w1_basis", [token_ffn_basis_num, d_model, d_ff],
                                   initializer=tf.random_normal_initializer(stddev=0.01))
        w2_basis = tf.get_variable("w2_basis", [token_ffn_basis_num, d_ff, d_model],
                                   initializer=tf.random_normal_initializer(stddev=0.01))

        # 每个 token 的组合系数 [l, token_ffn_basis_num]
        token_coef1 = tf.get_variable("token_coef1", [l, token_ffn_basis_num],
                                      initializer=tf.random_normal_initializer(stddev=0.01))
        token_coef2 = tf.get_variable("token_coef2", [l, token_ffn_basis_num],
                                      initializer=tf.random_normal_initializer(stddev=0.01))

        token_coef1 = tf.nn.softmax(token_coef1, axis=-1)
        token_coef2 = tf.nn.softmax(token_coef2, axis=-1)

        # 加权组合 [l, d_model, d_ff]
        w1_weighted = tf.einsum('lb,bio->lio', token_coef1, w1_basis)
        w2_weighted = tf.einsum('lb,boj->loj', token_coef2, w2_basis)

        w1_weighted = tf.nn.l2_normalize(w1_weighted, axis=1)
        w2_weighted = tf.nn.l2_normalize(w2_weighted, axis=1)

        # FFN 计算 (无门控)
        h = tf.einsum('bli,lio->blo', hidden_states, w1_weighted)

        # 每个 token 独立的 scale: [l, d_ff]
        scale = ngpt_scale("scale", [l, d_ff], scale_init=1.0,
                           scale_alpha=np.sqrt(d_model))

        h = tf.nn.l2_normalize(h, axis=-1)
        h = scale * h
        h = swish(h)

        out = tf.einsum('blo,loj->blj', h, w2_weighted)
        out = tf.nn.l2_normalize(out, axis=-1)

        return out


def normalized_ffn_swish_glu_basis_weighted(hidden_states, d_model, d_ff, token_ffn_basis_num,
                                            scope="normalized_ffn_swish_glu_basis_weighted",
                                            reuse=None):
    """
    基于基函数加权组合的 SwiGLU 归一化 FFN。

    结合 SwiGLU 门控结构与基函数加权机制，每个 token 通过独立的组合系数
    从共享的基函数矩阵中构建个性化权重，实现细粒度自适应变换。

    Args:
        hidden_states: [B, l, d_model]，l 为 token 数量。
        d_model: 输入/输出维度。
        d_ff: FFN 中间层维度。
        token_ffn_basis_num: 基础权重矩阵的数量。
        scope: variable_scope 名称。
        reuse: 是否复用变量。

    Returns:
        out: [B, l, d_model]，已 L2 归一化。
    """
    with tf.variable_scope(scope, reuse=reuse):
        l = hidden_states.get_shape().as_list()[1]

        # 基础权重矩阵 (三组，对应 SwiGLU 的 u、v 和输出投影)
        w1_basis = tf.get_variable("w1_basis", [token_ffn_basis_num, d_model, d_ff],
                                   initializer=tf.random_normal_initializer(stddev=0.01))
        w2_basis = tf.get_variable("w2_basis", [token_ffn_basis_num, d_model, d_ff],
                                   initializer=tf.random_normal_initializer(stddev=0.01))
        w3_basis = tf.get_variable("w3_basis", [token_ffn_basis_num, d_ff, d_model],
                                   initializer=tf.random_normal_initializer(stddev=0.01))

        # 每个 token 的组合系数 [l, token_ffn_basis_num]
        token_coef1 = tf.get_variable("token_coef1", [l, token_ffn_basis_num],
                                      initializer=tf.random_normal_initializer(stddev=0.01))
        token_coef2 = tf.get_variable("token_coef2", [l, token_ffn_basis_num],
                                      initializer=tf.random_normal_initializer(stddev=0.01))
        token_coef3 = tf.get_variable("token_coef3", [l, token_ffn_basis_num],
                                      initializer=tf.random_normal_initializer(stddev=0.01))

        token_coef1 = tf.nn.softmax(token_coef1, axis=-1)
        token_coef2 = tf.nn.softmax(token_coef2, axis=-1)
        token_coef3 = tf.nn.softmax(token_coef3, axis=-1)

        # 加权组合 [l, d_model, d_ff] 或 [l, d_ff, d_model]
        w1_weighted = tf.einsum('lb,bio->lio', token_coef1, w1_basis)
        w2_weighted = tf.einsum('lb,bio->lio', token_coef2, w2_basis)
        w3_weighted = tf.einsum('lb,boj->loj', token_coef3, w3_basis)

        # 沿嵌入维度归一化
        w1_weighted = tf.nn.l2_normalize(w1_weighted, axis=1)
        w2_weighted = tf.nn.l2_normalize(w2_weighted, axis=1)
        w3_weighted = tf.nn.l2_normalize(w3_weighted, axis=1)

        # SwiGLU 计算: u = x @ w1, v = x @ w2
        u = tf.einsum('bli,lio->blo', hidden_states, w1_weighted)
        v = tf.einsum('bli,lio->blo', hidden_states, w2_weighted)

        # 每个 token 独立的 scale: [l, d_ff]
        u_scale = ngpt_scale("u_scale", [l, d_ff], scale_init=1.0, scale_alpha=1.0)
        v_scale = ngpt_scale("v_scale", [l, d_ff], scale_init=1.0,
                             scale_alpha=np.sqrt(d_model))

        scaled_u = u_scale * u
        scaled_v = v_scale * v

        # SwiGLU: h = swish(v) * u
        h = swish(scaled_v) * scaled_u

        # 输出投影: out = h @ w3
        out = tf.einsum('blo,loj->blj', h, w3_weighted)
        out = tf.nn.l2_normalize(out, axis=-1)
        return out

def generailized_unimixer_block(x, block_size, token_num_after_reshape, seq_ffn_basis_num, token_ffn_basis_num):
    """
    广义的 UniMixer 块：局部投影 + 基函数加权全局 FFN。
    Args:
        x: [B, emb_dim]，其中 emb_dim 必须能被 block_size 整除。
        block_size: 每个 block 的维度 K。
        token_num_after_reshape: 全局 FFN 的中间维度。
        seq_ffn_basis_num: 局部基矩阵数量。
        token_ffn_basis_num: 全局 FFN 基函数数量。
    Returns:
        x_global: [B, emb_dim]，已 L2 归一化。
    """
    emb_dim = x.get_shape().as_list()[1]
    if emb_dim % block_size != 0:
        raise ValueError(
            f"emb_dim ({emb_dim}) must be divisible by block_size ({block_size})"
        )

    N = emb_dim // block_size
    K = block_size

    W_1 = tf.get_variable('W_1', shape=[N, seq_ffn_basis_num])
    W_V = tf.get_variable('v_basis_matrix', shape=(seq_ffn_basis_num, K, K))

    W_local = tf.einsum('nk,kde->nde', W_1, W_V)
    nW_local = tf.nn.l2_normalize(W_local, axis=1)
    x_reshaped = tf.reshape(x, [-1, N, K])
    x_reshaped = tf.nn.l2_normalize(x_reshaped, axis=-1)

    x_local = tf.einsum('bnk,nkh->bnh', x_reshaped, nW_local)
    
    x_local_trans = tf.transpose(x_local, perm=[0, 2, 1])
    x_local_trans = tf.nn.l2_normalize(x_local_trans, axis=-1)

    x_global = normalized_ffn_no_glu_basis_weighted(
        x_local_trans, N, token_num_after_reshape, token_ffn_basis_num)

    x_global = tf.transpose(x_global, perm=[0, 2, 1])
    x_global = tf.nn.l2_normalize(x_global, axis=-1)
    return x_global

def UniMixer(hidden_states, d_model, block_size, token_num_after_reshape, seq_ffn_basis_num, token_ffn_basis_num, d_ff,
                         scope="UniMixer", is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        l = hidden_states.get_shape().as_list()[1]

        # ---- Attention-like Mixing (UniMixer) ----
        mixing_h = tf.reshape(hidden_states, [-1, l * d_model])

        mixing_h = generailized_unimixer_block(
            mixing_h, block_size, token_num_after_reshape, seq_ffn_basis_num, token_ffn_basis_num
        )

        mixing_h = tf.reshape(mixing_h, [-1, l, d_model])
        mixing_h = tf.nn.l2_normalize(mixing_h, axis=-1)

        mixing_h = normalized_residual_update(
            hidden_states, mixing_h, scope="alpha1", alpha_init=0.05
        )

        # ---- FFN (per-token-weight SwiGLU, parallel via einsum) ----
        ffn_output = normalized_ffn_swish_glu_basis_weighted(
            mixing_h, d_model, d_ff, token_ffn_basis_num, scope="Token_FFN", reuse=reuse
        )

        out = normalized_residual_update(
            mixing_h, ffn_output, scope="alpha2", alpha_init=0.05)

        return out

class MixerConfig:
    def __init__(self):
        self.token_num = 32 #origin = 32
        self.d_kv = 128 #origin = 512 每个head的维度，head数量等于token数量
        self.num_block = 2
        self.block_size = 32
        self.seq_ffn_basis_num = 4
        self.token_ffn_basis_num = 8
        self.token_num_after_reshape = (self.token_num * self.d_kv) // self.block_size
        self.d_ff_coefficient = 2

mixer_config = MixerConfig()

# hidden_states = tf.stack(all_processed_tokens, axis=1)
hidden_states = None # hidden_states: [B, token_num, d_kv] 这个是输入的token嵌入

with tf.variable_scope("RankMixer_backbone"):
        hidden_states = tf.nn.l2_normalize(hidden_states, axis=-1)
        mixing_ffn_dim = d_kv * mixer_config.d_ff_coefficient

        for i in range(mixer_config.num_block):
            hidden_states = UniMixer(
                hidden_states,
                d_model = mixer_config.d_kv,
                block_size = mixer_config.block_size,
                token_num_after_reshape = mixer_config.token_num_after_reshape, 
                seq_ffn_basis_num = mixer_config.seq_ffn_basis_num,
                token_ffn_basis_num = mixer_config.token_ffn_basis_num,
                d_ff = mixing_ffn_dim,
                scope=f"RankMixer_block_{i}",
                is_training=is_training
            )

        hidden_states = rms_norm(hidden_states, scope=f"final_fusion_norm_{dim_index}")
        rankmixer_output = tf.reduce_mean(hidden_states, axis=1, name=f"rankmixer_mean_pooling_output_{dim_index}_{token_num}_{d_kv}")