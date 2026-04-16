from __future__ import print_function
import numpy as np
import math
import os
import argparse
import sys
from tensorflow.keras.backend import expand_dims, repeat_elements
from tensorflow.keras.backend import sum as backend_sum
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

CUSTOM_OPT_SETS = {}

training = args.mode == 'train'
random_seed = 666
use_print = True
step=config.get_step()
print_tag = '#==#'
tf.random.set_random_seed(random_seed)
def print_tensor(name, x, num_steps=100, num_sample=10):
    """
    print tensor
    """
    extra_ops = []
    # 1. compute matrix mean and var
    if use_print and training:
        m, v = tf.nn.moments( tf.reshape(x, [-1]), [0])
        # 2. compute zero rate
        zero_rate = 1 - tf.math.count_nonzero(x) / tf.size(x, out_type=tf.int64)
        extra_ops.append(tf.cond(tf.equal(tf.mod(step, num_steps), 1), lambda: tf.print(print_tag,
                                  "step=", step, ", ", "name=", name, ", ", "shape=", tf.shape(x), ", ",
                                  "min=", tf.reduce_min(x), ", ", "max=", tf.reduce_max(x), ", ", "mean=", m, ", ",
                                  "var=", v, ", ", "zero_rate=", zero_rate, ", ", "sample=", tf.strings.reduce_join(tf.strings.as_string(tf.reshape(x[:num_sample], [-1])[:num_sample]), separator="_"),
                                  print_tag, sep=""), lambda: tf.no_op()))
    return extra_ops
def log1p(feat):
    return tf.math.sign(feat) * tf.math.log1p(tf.abs(feat))

if args.mode == 'train':
    from kai.tensorflow.utils import data_table
    class DumpTensorHook(config.training.RunHookBase):
        
        def __init__(self, table_name, dump_tensors_dict):
            """
                本Hook用于获取tf图中dump_tensors_dict对应的tensor数据，导出到HDFS上
            Args:
                table_name (string): 表名
                dump_tensors_dict (dict): 需要导出的tensor数据，dict(tensor_name, tensor_op)
            """
            assert isinstance(dump_tensors_dict, dict)
            worker_id = config.current_rank()
            model_path = config.Config().save_option.model_path
            # 新建一个表
            self._dump_table = data_table.DataTable(
                table_name=table_name, worker_id=worker_id, model_path=model_path, keep_num=-1)
            self._dump_tensors_dict = dump_tensors_dict

        def before_step_run(self, step_run_context):
            """
                将 self._dump_tensors_dict 中的tensor注入fetches中
                后续step run图时会自动跑出来对应Tensor的数值

            Args:
                step_run_context (_type_): _description_

            Returns:
                _type_: _description_
            """
            return config.training.StepRunArgs(fetches=self._dump_tensors_dict)

        def after_step_run(self, step_run_context, step_run_values):
            """
                获取run图的结果，将结果写入表中

            Args:
                step_run_context (_type_): _description_
                step_run_values (_type_): _description_
            """
            sink_data = {}
            for name, op in self._dump_tensors_dict.items():
                value = step_run_values.result[name]
                batch_size = value.shape[0]
                sink_data[name] = value.reshape(batch_size, -1)

            step_id = step_run_context.descr_list.step
            pass_id = step_run_context.descr_list.pass_id
            sink_data["step_id"] = [step_id] * batch_size
            sink_data["pass_id"] = [pass_id] * batch_size
            self._dump_table.append_batch(sink_data)

def cache_embedding(embedding, slot, dim, name):
    global CUSTOM_OPT_SETS
    param_attr = config.nn.ParamAttr(
            initializer=config.nn.ConstInitializer(value=0.0),
        )
    cache_embedding = config.new_embedding(name, dim=dim, slots=[slot], param_attr=param_attr)
    if args.mode == 'train':
        CUSTOM_OPT_SETS[cache_embedding] = {
            "opt_type": "AssignAdd",
            "decay_rate": 0.0,
            "add_rate": 1.0,
            "grad": embedding
        }
        return embedding
    else:
        return cache_embedding
# get labels
all_labels = ['retention_label', 'lt7', 'pred_lt30', 'lt30', 'app_usage_duration_0d_bin','app_usage_duration_1d_bin','app_usage_duration_2d_bin', 'valid_duration_lt7', 'is_not_0vv_lt7', 'valid_retention_label',
                         'app_usage_duration_3d_bin', 'app_usage_duration_4d_bin', 'app_usage_duration_5d_bin', 'app_usage_duration_6d_bin', 'app_usage_duration_7d_bin', 'valid_lt7', 'fusion_prtr']

def get_op(config, embeddings, op_name, per_step):
    embedding_means = [tf.nn.moments(embedding, 0)[0] for embedding in embeddings]
    my_step = config.get_step()
    op = tf.cond(
        tf.equal(tf.mod(my_step, per_step), 1),
        lambda: tf.print("step",
                         my_step,
                         op_name,
                         embedding_means,
                         summarize=-1,
                         output_stream=sys.stdout), lambda: tf.no_op())
    return [op]
    
def clip_to_boolean(tensor):
    return tf.clip_by_value(tensor, 0.0, 1.0)

def get_label(name):
    assert name in all_labels, name
    # config.get_label(name): send_to_mio (label_attr)
    # config.get_extra_param(name): send_to_mio(attrs)
    # return config.get_extra_param(name)
    return config.get_label(name)

def get_predict_param():
    param = {}
    if args.mode != "train":
        param["compress_group"]="USER"
    return param

def mio_dense_layer(inputs, units, activation, name):
    # tf.Dense-like layer similar to that of mio-dnn
    if not isinstance(inputs, list):
        inputs = [inputs]
    weight_name = name+'_kernel'
    assert (len(inputs) > 0)
    with tf.name_scope(name):
        i = inputs[0]
        weight = tf.get_variable(weight_name, (i.get_shape()[1], units))

        o = tf.matmul(i, weight, name=f"{name}_mul")

        for idx, extra_i in enumerate(inputs[1:]):
            weight = tf.get_variable(f"{weight_name}_extra_{idx}", (extra_i.get_shape()[1], units))
            o += tf.matmul(extra_i, weight, name=f"{name}_extra_mul_{idx}")

        bias = tf.get_variable(f"{weight_name}_bias", (units))
        o = tf.nn.bias_add(o, bias)

        if activation is not None:
            o = activation(o)

        return o
    
def mio_dense_layer_with_bn(inputs, units, activation, batch_norm, name, training=False):
    # tf.Dense-like layer similar to that of mio-dnn
    if not isinstance(inputs, list):
        inputs = [inputs]
    weight_name = name+'_kernel'
    assert (len(inputs) > 0)
    with tf.name_scope(name):
        i = inputs[0]
        weight = tf.get_variable(weight_name, (i.get_shape()[1], units))

        o = tf.matmul(i, weight, name=f"{name}_mul")

        for idx, extra_i in enumerate(inputs[1:]):
            weight = tf.get_variable(f"{weight_name}_extra_{idx}", (extra_i.get_shape()[1], units))
            o += tf.matmul(extra_i, weight, name=f"{name}_extra_mul_{idx}")

        bias = tf.get_variable(f"{weight_name}_bias", (units))
        o = tf.nn.bias_add(o, bias)

        if batch_norm == True:
            o = tf.layers.batch_normalization(o, training=training, name=f"{name}_batch_norm")

        if activation is not None:
            o = activation(o)

        return o


def simple_dense_network(inputs, units, name, act=tf.nn.relu, dropout = 0.0):
    output = inputs
    for i, unit in enumerate(units):
        output = mio_dense_layer(output, unit, act, name='dense_{}_{}'.format(name, i))
        output = tf.layers.dropout(output, rate=dropout*bool(args.mode == 'train'))
    return output

def MMOE(inputs, task_names, expert_nuits, tower_units, num_experts):
    with tf.variable_scope('mmoe', reuse=tf.AUTO_REUSE):
        expert_list = []
        for idx in range(num_experts):
            expert_output = simple_dense_network(inputs, expert_nuits, "expert_%i" %idx, act=tf.nn.relu, dropout=0.1)
            expert_list.append(expert_output)
        expert = tf.stack(expert_list, axis=1)
        outputs = []
        for task_idx, task_name in enumerate(task_names):
            gate = simple_dense_network(inputs, [num_experts], "gate_task_%s" %task_name, act=tf.nn.softmax, dropout=0.1)
            gate = tf.reshape(gate, [-1, 1, num_experts])
            mmoe_output = tf.reshape(tf.matmul(gate, expert), [-1,expert_nuits[-1]])
            mmoe_output = simple_dense_network(mmoe_output, tower_units, "tower_task_%s" %task_name, act=tf.nn.relu, dropout=0.1)
            outputs.append(mmoe_output)
        return outputs

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

def ffn_swish_glu(hidden_states,
                  d_model,
                  d_ff,
                  scope="ffn_swish_glu",
                  reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        d_ff_glu = int(d_ff * 2 / 3)
        
        if d_ff % 3 != 0:
            d_ff_glu = math.ceil(d_ff * 2 / 3 / 2) * 2

        reshaped_input = tf.squeeze(hidden_states, axis=1)

        gate = mio_dense_layer(reshaped_input, d_ff_glu, activation=None, name='gate_proj')
        up = mio_dense_layer(reshaped_input, d_ff_glu, activation=None, name='up_proj')

        activated_up = swish(gate) * up

        output = mio_dense_layer(activated_up, d_model, activation=None, name='down_proj')
        
        output_states = tf.expand_dims(output, axis=1)
        return output_states

def _noisy_top_k_gating(
        inputs,           # [B, T, d_model]
        num_experts,
        k,
        d_model,
        seq_len,
        noise_coef=1.0,
        is_training=True,
        scope="noisy_gate"):
    """
    返回：
        gates       : [B, T, num_experts]  稀疏 softmax，仅 top-k 非零
        indices     : [B, T, k]            被选中的专家 id
        load_balance_loss: 标量 Tensor，供优化器使用
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # shape = tf.shape(inputs)
        # batch, seq_len = shape[0], shape[1]
        # d_model = inputs.get_shape().as_list()[-1]  # 静态维度

        x = tf.reshape(inputs, [-1, d_model])  # [B*T, d]

        # 线性映射得到 logits
        # gate_logits = tf.layers.dense(x, num_experts, use_bias=False, name="gate")   # [B*T, E]
        gate_logits = mio_dense_layer(x, num_experts, activation=None, name="gate")

        # 噪声分支
        noise_logits = mio_dense_layer(x, num_experts, activation=None, name="noise")
        noise_std = tf.nn.softplus(noise_logits)            # 保证 >0
        eps = tf.random_normal(tf.shape(gate_logits))
        if is_training:
            gate_logits = gate_logits + noise_coef * eps * noise_std


        # Top-k
        top_logits, indices = tf.nn.top_k(gate_logits, k=k, sorted=True)  # 保证顺序

        expert_range = tf.range(num_experts, dtype=indices.dtype) # 形状: [num_experts]
        one_hot_like_indices_bool = tf.math.equal(
            tf.expand_dims(indices, axis=-1), # 扩展后形状: [B*T, k, 1]
            expert_range                      # 广播后形状: [1, 1, num_experts]
        ) # 比较结果形状: [B*T, k, num_experts]
        one_hot_indices = tf.cast(one_hot_like_indices_bool, dtype=tf.float32)

        top_k_mask = tf.reduce_sum(one_hot_indices, axis=1)
        # very_negative = tf.constant(-1e9, dtype=tf.float32)
        very_negative = -1e9 
        masked_logits = gate_logits * top_k_mask + (1.0 - top_k_mask) * very_negative

        # Softmax on masked logits
        gates = tf.nn.softmax(masked_logits, axis=-1)  # [B*T, E]

        # Load balance loss
        prob_sum = tf.reduce_mean(gates, axis=0)  # [E]
        total_tokens = tf.cast(tf.shape(indices)[0], tf.float32)  # B*T
        expert_assignments = tf.reduce_sum(one_hot_indices, axis=[0, 1])
        
        f = expert_assignments / (total_tokens * k)  # [E]
        load_balance_loss = tf.reduce_sum(f * prob_sum) * float(num_experts)

        # Reshape back
        gates = tf.reshape(gates, [-1, seq_len, num_experts])
        indices = tf.reshape(indices, [-1, seq_len, k])

    return gates, indices, load_balance_loss

def SparseMoe_NoNorm(seq_len, token_input, # 输入形状应为 [B, 1, D]
                     d_model,
                     d_ff,
                     expert_num,
                     k,
                     scope="SparseMoe",
                     denseFFN=False,
                     is_training=True,
                     reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        gate, indices, load_balance_loss = _noisy_top_k_gating(token_input, expert_num, k, d_model, seq_len, is_training=is_training)
        
        if denseFFN:
            gate = tf.ones_like(gate)
        
    #   output = tf.zeros_like(token_input)
        output = tf.zeros(tf.shape(token_input), dtype=token_input.dtype)
        gate_splits = tf.split(gate, num_or_size_splits=expert_num, axis=2)
        for i in range(expert_num):
            expert_output = ffn_swish_glu(token_input, d_model, d_ff, f"_ffn{i}_")
            weighted_expert_output = gate_splits[i] * expert_output
            output += weighted_expert_output

        return gate, output, load_balance_loss

def manual_logsumexp_v2(logits, axis=2, keepdims=True):
    # 1. 显式强制转换精度，防止 float16 带来的精度崩塌
    original_dtype = logits.dtype
    logits = tf.cast(logits, tf.float32)
    
    # 2. 提取最大值
    logits_max = tf.reduce_max(logits, axis=axis, keepdims=True)
    
    # 3. 稳定计算
    centered_logits = logits - logits_max
    exp_logits = tf.exp(centered_logits)
    sum_exp = tf.reduce_sum(exp_logits, axis=axis, keepdims=keepdims)
    
    # 4. 增加 epsilon 避免 log(0)
    log_sum = tf.log(sum_exp + 1e-12)
    
    # 5. 确保 logits_max 的维度与 log_sum 对齐
    if not keepdims:
        logits_max = tf.squeeze(logits_max, axis=axis)
        
    result = log_sum + logits_max
    return tf.cast(result, original_dtype)

def log_sinkhorn(logits, n_iters=20, temperature=0.1):
    original_shape = logits.get_shape().as_list()
    # 安全地将输入统一为 [B, K, K] 格式：
    # - 若输入为 [K, K] (rank=2) → reshape 为 [1, K, K]
    # - 若输入为 [N, K, K] (rank=3) → 保持 [N, K, K]
    # - reshape 自动推导 batch 维（-1），无需条件判断
    a = logits.get_shape().as_list()[-2]
    b = logits.get_shape().as_list()[-1]
    logits_3d = tf.reshape(logits, [-1, a, b])  # 关键：消除 tf.cond
    
    logits_3d = logits_3d / temperature

    if is_training:
        for _ in range(n_iters):
            logits_3d = logits_3d - manual_logsumexp_v2(logits_3d, axis=2, keepdims=True)
            logits_3d = logits_3d - manual_logsumexp_v2(logits_3d, axis=1, keepdims=True)
    
    res_3d = tf.exp(logits_3d)
    return tf.reshape(res_3d, original_shape)  # 恢复原始形状

def efficient_kron_interaction_doubly_stochastic(x, block_size, r, basis_num, sinkhorn_iters=1):
    emb_dim = x.get_shape().as_list()[1]
    if emb_dim % block_size != 0:
        raise ValueError(f"emb_dim ({emb_dim}) must be divisible by block_size ({block_size})")
    
    N = emb_dim // block_size          # Chunks
    K = block_size         # Chunk dim

    W_A =  tf.get_variable(
        "Matrix_WA_logits", 
        shape=[N, r],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )
    W_B =  tf.get_variable(
        "Matrix_WB_logits", 
        shape=[r, N],
        initializer=tf.random_normal_initializer(stddev=0.1)
    )
    A_logits = W_A @ W_B

    W_1 = tf.get_variable('W_1', shape=[N, basis_num])
    W_V = tf.get_variable('v_basis_matrix', shape=(basis_num, K, K))
    W_logits = tf.einsum('tk,kde->tde', W_1, W_V) 

    #————————————————————全局/局部参数矩阵的对称约束————————————————————
    # A_logits_constraint = (A_logits + tf.transpose(A_logits, perm=[1, 0])) * 0.5
    # W_logits_constraint = (W_logits + tf.transpose(W_logits, perm=[0, 2, 1])) * 0.5

    A = log_sinkhorn(A_logits, n_iters=sinkhorn_iters, temperature=0.05)
    W = log_sinkhorn(W_logits, n_iters=sinkhorn_iters, temperature=0.05)

    #————————————————————A，W收敛监控开始————————————————————
    A_entropy = -tf.reduce_mean(tf.reduce_sum(A * tf.math.log(A + 1e-10), axis=-1)) # 行熵  
    tf.summary.scalar("A_row_entropy", A_entropy)
    # row_max衡量每行最大的那个值有多大 → 从1/N（均匀）到接近1（收敛）
    tf.summary.scalar("A_row_max_mean", tf.reduce_mean(tf.reduce_max(A, axis=-1)))
    W_entropy = -tf.reduce_mean(tf.reduce_sum(W * tf.math.log(W + 1e-10), axis=-1))
    tf.summary.scalar("W_row_entropy", W_entropy)
    tf.summary.scalar("W_row_max_mean", tf.reduce_mean(tf.reduce_max(W, axis=-1)))
    #————————————————————A，W收敛监控结束————————————————————

    x_reshaped = tf.reshape(x, [-1, N, K])
    x_local = tf.einsum('bni, nio -> bno', x_reshaped, W)
    x_global = tf.einsum('mn, bno -> bmo', A, x_local)

    # need_print_tensor_dict['r'] = r
    # need_print_tensor_dict['basis_num'] = basis_num

    output = tf.reshape(x_global, [-1, emb_dim])
    return output

def UniMixer(hidden_states_x, hidden_states_y, d_model, block_size, r, basis_num, d_ff, expert_num, k, scope="UniMixer", denseFFN=False, is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        b, l, d = hidden_states_x.get_shape().as_list()

        normed_y_mix = rms_norm(hidden_states_y, scope="siamese_norm_pre_mixing")
        mixing_input = hidden_states_x + normed_y_mix

        # ------------------------------UniMixer Start----------------------------------
        # 转置
        mixing_h = tf.transpose(mixing_input, perm=[0, 2, 1])
        mixing_h = tf.reshape(mixing_h, [-1, l * d_model])

        # mixing_h = tf.reshape(hidden_states, [-1, l * d_model])
        mixing_h = efficient_kron_interaction_doubly_stochastic(mixing_h, block_size, r, basis_num)
        mixing_h = tf.reshape(mixing_h, [-1, l, d_model])

        pffn_input = rms_norm(mixing_input + mixing_h, scope="post_mixing_norm")
        # ------------------------------UniMixer End----------------------------------
        # mixed_hidden_states = rms_norm(hidden_states + mixing_h, scope="post_mixing_norm")
        
        all_token_outputs = []
        all_gate_outputs = []
        total_load_balance_loss = 0.0

        for i in range(l):
            token_slice = tf.slice(pffn_input, [0, i, 0], [-1, 1, -1])
            with tf.variable_scope(f"Token_{i}_MoE"):
                gate, token_output, load_balance_loss = SparseMoe_NoNorm(1, token_slice, d_model, d_ff, expert_num, k, is_training=is_training)
            
            all_token_outputs.append(token_output)
            all_gate_outputs.append(gate)
            total_load_balance_loss += load_balance_loss

        moe_output = tf.concat(all_token_outputs, axis=1)
        final_gates = tf.concat(all_gate_outputs, axis=1)
        avg_load_balance_loss = total_load_balance_loss / l
        
        final_x = rms_norm(hidden_states_x + moe_output, scope="post_norm_moe")
        final_y = hidden_states_y + moe_output

        return final_gates, final_x, final_y, avg_load_balance_loss


def order_regression(input_embedding,
                     dnn_layers=[16],
                     cond_probs=6,
                     scope='default',
                     residual_logit=None):
    pred_cond_probs_dict = {}
    pred_cut_probs_dict = {}
    pred = 1
    with tf.variable_scope("order_regression_{}".format(scope)):
        for i in range(cond_probs):
            _name = "{}_{}".format(i + 2, i + 1)
            out = simple_dense_network(input_embedding, dnn_layers, name="cond_prob_{}".format(_name))
            out = mio_dense_layer(out, 1, activation=None, name="cond_prob_{}_last_fc".format(_name))
            if residual_logit is not None:
                pred_cond_probs_dict[_name] = tf.nn.sigmoid(out+residual_logit[i + 1])
            else:
                pred_cond_probs_dict[_name] = tf.nn.sigmoid(out)
        for i in range(cond_probs):
            _last_name = "{}".format(i + 1)
            _cur_name = "{}".format(i + 2)
            _cond_name = '{}_{}'.format(i + 2, i + 1)
            if i == 0:
                pred_cut_probs_dict[_cur_name] = pred_cond_probs_dict[_cond_name]
            else:
                pred_cut_probs_dict[_cur_name] = pred_cond_probs_dict[_cond_name] * pred_cut_probs_dict[_last_name]
            pred += pred_cut_probs_dict[_cur_name]
        # pred += pred_cut_probs_dict[_cur_name] * 6
    return {
        "cond_prob": pred_cond_probs_dict,
        "cut_prob": pred_cut_probs_dict,
        "pred": pred
    }

def new_expand_emb(fea_name, dim, expand, slot_id):
    assert expand > 1
    assert isinstance(slot_id, int)
    assert dim >= 1
    action_list = config.new_embedding(fea_name, dim=dim, slots=[slot_id], expand=expand, **get_predict_param())
    action_list = tf.reshape(action_list, [-1, expand, dim])
    return action_list
    


def fix_prob(x):
    x = x + 1e-9
    _sum = tf.reduce_sum(x, axis=1, keepdims=True)
    x = x / _sum
    x = tf.clip_by_value(x, 1e-9, 1.0)

    return x

def kl_divergence_loss(x, y):
    x = fix_prob(x)
    y = fix_prob(y)
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)

    return tf.distributions.kl_divergence(X, Y)

class MixerConfig:
    def __init__(self):
        self.num_block = 4 #层数
        self.d_ff_coefficient = 2 # ffn中间层维度倍数，默认2倍
        self.expert_num = 1 #专家数量
        self.k = 1 #TopK
        self.alpha = 0.0 #负载均衡损失函数权重

        self.token_num_list = [16]
        self.d_kv_list = [96]

        self.block_size = 4
        self.r = (16 * 96) // (4 * 8) # 1536/8
        self.basis_num = 4 #3
mixer_config = MixerConfig()
is_training = (args.mode == 'train')
# global variables
ops = []
# id features
input_emb_dim = 4
wide_emb_dim = 4
request_feature_slots = [56, 58]
tense_slots = [61, 62, 63, 64, 65, 66, 161, 162, 163, 254]
user_common_slots = [50, 51, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115] + \
                [255,256,258,259,260,270,271,272,273,274,275,
                276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 
                303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 
                328, 329, 330, 331, 332, 333, 334, 335,336,337,338,339,340,341,
                342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
                116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,143,144,145]
rtb_feature_slots = [53, 54, 55, 57, 59, 100, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                     210, 211, 212, 213, 214, 215, 216, 217, 218,219,220, 221, 222, 223, 224, 225, 226, 227, 228,
                     229, 230, 231, 232, 233, 234, 235, 236,237,238,239, 240, 241, 242, 243, 244, 245, 246, 247,
                     248, 249, 250, 251, 252, 253]
rtb_feature_slots_item = [2, 3, 5, 52, 60]

user_common_embs = config.new_embedding("user_common_input", dim=input_emb_dim, slots=user_common_slots,**get_predict_param())
request_feature_embs = config.new_embedding("request_feature_embs", dim=input_emb_dim, slots=request_feature_slots,**get_predict_param())
tense_embs = config.new_embedding("tense_embs", dim=input_emb_dim, slots=tense_slots,**get_predict_param())
active_app_emb = config.new_embedding("active_app_emb", dim=16, slots=[150,151,82,85,164,165], **get_predict_param())
install_app_emb = config.new_embedding("install_app_emb", dim=16, slots=[152,153,154,155,156,157,158,159], **get_predict_param())
colossus_lt_emb = config.new_embedding("colossus_lt_emb", dim=16, slots=[361,362,363,364,365,366], **get_predict_param())
channel_mix = config.new_embedding("channel_mix", dim=4, slots=[13, 90, 160])
usergroup_mix_embedding =  config.get_extra_param("usergroup_mix", size=4, **get_predict_param())
colossus_feature_embedding =  config.get_extra_param("colossus_feature", size=12, **get_predict_param())

# only rtb
rtb_feature = config.new_embedding("rtb_feature", dim=input_emb_dim, slots=rtb_feature_slots,**get_predict_param())
rate_mix_embedding =  config.get_extra_param("rate_mix", size=54, **get_predict_param())

mmoe_input = tf.concat([user_common_embs, request_feature_embs, tense_embs, active_app_emb, install_app_emb,colossus_lt_emb,
                    channel_mix, usergroup_mix_embedding, colossus_feature_embedding], axis=1)
promotion_id_emb = config.new_embedding("promotion_id_emb", dim=input_emb_dim, slots=[7])
channel_type_emb = config.new_embedding("channel_type_emb", dim=input_emb_dim, slots=[8])
# creative_id_emb = config.new_embedding("creative_id_emb", dim=input_emb_dim, slots=[6])
rtb_feature_item = config.new_embedding("rtb_feature_item", dim=input_emb_dim, slots=rtb_feature_slots_item)

gdt_embedding =  config.get_extra_param("gdt_emb_fusion", size=32, **get_predict_param())

feature_list = tf.concat([mmoe_input, promotion_id_emb, channel_type_emb, rate_mix_embedding, rtb_feature, rtb_feature_item, gdt_embedding], axis=1)
final_concatenated_features = tf.concat(feature_list, axis=1)

with tf.variable_scope("RankMixer_backbone"):
    total_input_dim = final_concatenated_features.get_shape()[-1]

    expert_num = mixer_config.expert_num
    mixer_regloss = 0
    rankmixer_output_list = []
    all_mixer_ops = []

    for dim_index in range(len(mixer_config.token_num_list)):
        token_num = mixer_config.token_num_list[dim_index]
        d_kv = mixer_config.d_kv_list[dim_index]

        remainder = total_input_dim % token_num
        if remainder != 0:
            padding_size = token_num - remainder
            batch_size = tf.shape(final_concatenated_features)[0]
            padding_tensor = tf.zeros([batch_size, padding_size], dtype=final_concatenated_features.dtype)
            padded_features = tf.concat([final_concatenated_features, padding_tensor], axis=1)
        else:
            padded_features = final_concatenated_features

        total_padded_dim = padded_features.get_shape().as_list()[-1]
        chunk_dim_per_token = total_padded_dim // token_num
        reshaped_features = tf.reshape(
            padded_features,
            [-1, token_num * chunk_dim_per_token]
        )

        token_slices_3d = tf.split(
            reshaped_features,
            num_or_size_splits=token_num,
            axis=1
        )  # [token_num, bs, token_dim]
        all_processed_tokens = []
        with tf.variable_scope(f"shared_token_projection_scope_{dim_index}_{token_num}_{d_kv}", reuse=tf.AUTO_REUSE):
            for i in range(len(token_slices_3d)):
            # for token_slice_2d in token_slices_3d:
                token_slice_2d = token_slices_3d[i]
                processed_slice = mio_dense_layer(
                    token_slice_2d,  # [bs, token_dim]
                    d_kv,
                    activation=None,
                    name=f"shared_token_projection_{dim_index}_{token_num}_{d_kv}"
                )
                all_processed_tokens.append(processed_slice)

        hidden_states = tf.stack(all_processed_tokens, axis=1)
        hidden_states_x = hidden_states  
        hidden_states_y = hidden_states

        mixing_ffn_dim = d_kv * mixer_config.d_ff_coefficient

        for i in range(mixer_config.num_block):
            gate, hidden_states_x, hidden_states_y, load_balance_loss = UniMixer(
                hidden_states_x,
                hidden_states_y,
                d_kv,
                mixer_config.block_size,
                mixer_config.r, 
                mixer_config.basis_num,
                mixing_ffn_dim,
                expert_num,
                mixer_config.k,
                scope=f"UniMixer_block_{i}_{dim_index}_{token_num}_{d_kv}",
                is_training=is_training
            )
            mixer_regloss += mixer_config.alpha * load_balance_loss

        final_normed_y = rms_norm(hidden_states_y, scope=f"final_fusion_norm_{dim_index}")
        final_output_tensor = hidden_states_x + final_normed_y
        rankmixer_output = tf.reduce_mean(final_output_tensor, axis=1, name=f"rankmixer_mean_pooling_output_{dim_index}_{token_num}_{d_kv}")

        rankmixer_output_list.append(rankmixer_output)

    rankmixer_output = tf.concat(rankmixer_output_list, axis=-1, name="rankmixer_output_concat")

with tf.name_scope('model'):
    # mmoe    
    task_names = ['rtr', 'lt7', 'duration', 'lt30_mu', 'lt30_sigma']
    expert_units, tower_units = [128, 64], [64, 32]

    p_rtr_tower = simple_dense_network(rankmixer_output, tower_units, 'rtr_tower')
    p_rtr_logit = mio_dense_layer(p_rtr_tower, 1, activation=None, name='rtr_logit')
    p_rtr = tf.nn.sigmoid(p_rtr_logit)
    
    lt7_preds = order_regression(rankmixer_output, dnn_layers=tower_units, scope='lt7', residual_logit=None)
    lt7_cond_prob_list = [lt7_preds['cond_prob']['{}_{}'.format(i + 2, i + 1)] for i in range(6)]
    lt7_prob_list = [lt7_preds['cut_prob']['{}'.format(i + 2)] for i in range(6)]
    lt7_pred = lt7_preds['pred']
    
    p_lt30gt1_tower = simple_dense_network(rankmixer_output, tower_units, 'lt30gt1_tower')
    p_lt30gt1_logit = mio_dense_layer(p_lt30gt1_tower, 1, activation=None, name='lt30gt1_logit')
    p_lt30gt1 = tf.nn.sigmoid(p_lt30gt1_logit)

    gwd_tower = simple_dense_network(rankmixer_output, tower_units, 'gwd_tower')
    gwd_logits = mio_dense_layer(gwd_tower, 2, activation=None, name='gwd_logits')
    p_gwd = tf.nn.softmax(gwd_logits)
    p_ngw = tf.slice(p_gwd, [0, 0], [-1, 1])
    p_gw = tf.slice(p_gwd, [0, 1], [-1, 1])

    gw_mu_tower = simple_dense_network(rankmixer_output, tower_units, 'gw_mu_tower')
    p_gw_mu = mio_dense_layer(gw_mu_tower, 1, activation=None, name='p_gw_mu')

    ngw_mu_tower = simple_dense_network(rankmixer_output, tower_units, 'ngw_mu_tower')
    p_ngw_mu = mio_dense_layer(ngw_mu_tower, 1, activation=None, name='p_ngw_mu')
    
    gw_sigma_tower = simple_dense_network(rankmixer_output, tower_units, 'gw_sigma_tower')
    p_gw_sigma = mio_dense_layer(gw_sigma_tower, 1, activation=tf.nn.softplus, name='p_gw_sigma')

    ngw_sigma_tower = simple_dense_network(rankmixer_output, tower_units, 'ngw_sigma_tower')
    p_ngw_sigma = mio_dense_layer(ngw_sigma_tower, 1, activation=tf.nn.softplus, name='p_ngw_sigma')

    p_lt30_mu = p_gw * p_gw_mu + p_ngw * p_ngw_mu
    p_lt30_sigma = p_gw * p_gw_sigma + p_ngw * p_ngw_sigma
    
    p_lt30 = tf.pow(1.6, p_lt30_mu + 0.5 * tf.square(p_lt30_sigma)) * p_lt30gt1 + 1.0
    
def to_weights(tensor):
    return tf.cast(tensor, tf.float32)

if args.mode == 'train':
    is_rtb = config.get_extra_param('is_rtb', size=1, reversed=False, default_value=0.0)
    is_rta = config.get_extra_param('is_rta', size=1, reversed=False, default_value=0.0)
    is_half_rtb = config.get_extra_param('is_half_rtb', size=1, reversed=False, default_value=0.0)
    install_flag = config.get_extra_param('install_flag', size=1, reversed=False, default_value=0.0)
    uninstall_flag = config.get_extra_param('uninstall_flag', size=1, reversed=False, default_value=0.0)
    is_0vv_flag = config.get_extra_param('is_0vv_flag', size=1, reversed=False, default_value=0.0)
    is_not_0vv_flag = config.get_extra_param('is_not_0vv_flag', size=1, reversed=False, default_value=0.0)
    lt7 = get_label('lt7') # 取值[1, 7]
    valid_lt7 = get_label('valid_lt7')
    pred_lt30 = get_label('pred_lt30')  # [1,30]
    pred_lt30_trans = pred_lt30 - 1.0  # [0, 29]
    lt30gt1_label = tf.cast(tf.where(pred_lt30_trans > 0, tf.ones_like(pred_lt30_trans), tf.zeros_like(pred_lt30_trans)), tf.float32)
    lt30 = get_label('lt30')
    valid_retention_label = get_label('valid_retention_label')
    fusion_prtr = get_label('fusion_prtr')
    valid_duration_lt7 = get_label('valid_duration_lt7')
    is_valid_lt30 = tf.cast(tf.where(pred_lt30 >= 1, tf.ones_like(pred_lt30), tf.zeros_like(pred_lt30)), tf.float32)
    is_valid_lt7 = tf.cast(tf.where(lt7 >= 1, tf.ones_like(lt7), tf.zeros_like(lt7)), tf.float32)
    is_not_0vv_lt7 = get_label('is_not_0vv_lt7')
    retention_label = clip_to_boolean(get_label('retention_label'))
    lost_180=config.get_extra_param('lost_180', size=1, reversed=False, default_value=0.0)
    is_gdt=config.get_extra_param('is_gdt', size=1, reversed=False, default_value=0.0)
    is_vivo=config.get_extra_param('is_vivo', size=1, reversed=False, default_value=0.0)
    is_oppo=config.get_extra_param('is_oppo', size=1, reversed=False, default_value=0.0)
    is_csj=config.get_extra_param('is_csj', size=1, reversed=False, default_value=0.0)
    is_qtt=config.get_extra_param('is_qtt', size=1, reversed=False, default_value=0.0)
    churn_gt_180 = config.get_extra_param('churn_gt_180', size=1, reversed=False, default_value=0.0)
    flag_colossus_exist = config.get_extra_param('flag_colossus_exist', size=1, reversed=False, default_value=0.0)
    duration_label = []
    for day_num in range(8):
        param_name =  f'app_usage_duration_{day_num}d_bin'  # 参数名格式如 'app_usage_duration_0d_bin'
        param_value = get_label(param_name)
        duration_label.append(param_value)
        
    rtr_targets = [
        ('rtr',     p_rtr,    retention_label,     tf.cast(tf.ones_like(is_rta), tf.float32),     'auc')
    ]
    q_name, preds, labels, weights, linear_regression = zip(*rtr_targets)
    rtr_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")

    # lt7
    lt7_num_of_class = 7
    lt7_onehot = tf.one_hot(tf.squeeze(tf.cast(valid_lt7 - 1, "int32"), -1), lt7_num_of_class)
    lt7_onehot = tf.cumsum(lt7_onehot, -1, reverse=True)
    lt7_cut_list = [] # 6个值 lt>=i+2, i=0,1,2,3,4,5   p(lt7>=k)的交叉熵损失的label
    for i in range(6):
        lt7_cut_list.append(tf.expand_dims(lt7_onehot[:,i+1], 1))
    lt7_cond_weight_list = [] # lt>=i+1, i=0,1,2,3,4,5 p(lt7>=k+1|lt7>=k)的交叉熵损失的weight
    for i in range(6):
        if i == 0:
            lt_weight_i = tf.ones_like(lt7_cut_list[0])
        else:
            lt_weight_i = tf.where(tf.equal(lt7_cut_list[i-1], 1), tf.ones_like(lt7_cut_list[i-1]), tf.zeros_like(lt7_cut_list[i-1])) #条件概率,计算i的时候,需要满足i-1是正样本
        lt7_cond_weight_list.append(lt_weight_i)
    
    lt7_target = []
    for i in range(6):
        lt7_target.append(("rta_lt7_cond_{}_{}".format(i+2, i+1), lt7_cond_prob_list[i], lt7_cut_list[i],
                           lt7_cond_weight_list[i] * is_rta * is_valid_lt7, "linear_regression"))
        lt7_target.append(("rtb_lt7_cond_{}_{}".format(i+2, i+1), lt7_cond_prob_list[i], lt7_cut_list[i],
                           lt7_cond_weight_list[i] * is_rtb * is_valid_lt7, "linear_regression"))
        lt7_target.append(("half_rtb_lt7_cond_{}_{}".format(i+2, i+1), lt7_cond_prob_list[i], lt7_cut_list[i],
                           lt7_cond_weight_list[i] * is_half_rtb * is_valid_lt7, "linear_regression"))
    lt7_prob_targets = []
    for i in range(6):
        lt7_prob_targets.append(("rta_lt7_leq_{}".format(i+2), lt7_prob_list[i], lt7_cut_list[i], is_rta * is_valid_lt7, "linear_regression"))
        lt7_prob_targets.append(("rtb_lt7_leq_{}".format(i+2), lt7_prob_list[i], lt7_cut_list[i], is_rtb * is_valid_lt7, "linear_regression"))
        lt7_prob_targets.append(("half_rtb_lt7_leq_{}".format(i+2), lt7_prob_list[i], lt7_cut_list[i], is_half_rtb * is_valid_lt7, "linear_regression"))

    lt7_target_for_loss = []
    for i in range(6):
        lt7_target_for_loss.append(("lt7_cond_{}_{}".format(i+2, i+1), lt7_cond_prob_list[i], lt7_cut_list[i],
                           lt7_cond_weight_list[i] * tf.cast(tf.ones_like(is_rta), tf.float32) * is_valid_lt7, "linear_regression"))
    q_name, preds, labels, weights, linear_regression = zip(*lt7_target_for_loss)
    lt7_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    
    # lt30gt1
    lt30gt1_targets = [
        ('lt30gt1',     p_lt30gt1,    lt30gt1_label,     tf.cast(tf.ones_like(is_rta), tf.float32) * is_valid_lt30,     'auc')
    ]
    q_name, preds, labels, weights, auc = zip(*lt30gt1_targets)
    lt30gt1_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")

    # lt30
    base = 1.6
    pred_lt30_safe = lt30gt1_label * pred_lt30_trans + (1 - lt30gt1_label) * tf.ones_like(pred_lt30_trans)
    log_pred_lt30 = tf.math.log(pred_lt30_safe) / tf.math.log(tf.constant(base, dtype=tf.float32))
    log_diff = log_pred_lt30 - p_lt30_mu  
    penalty = 0.5 * tf.square(p_lt30_sigma)
    log_normal_loss = 0.5 * (tf.square(log_diff) / tf.square(p_lt30_sigma) + tf.math.log(p_lt30_sigma)) + penalty
    log_normal_loss = tf.reduce_sum(log_normal_loss * lt30gt1_label * is_valid_lt30)

    # gw detect
    gw_thres = 3.0
    preds_gw_ptr = p_lt30gt1 * p_gw  # [batch_size, 1]
    preds_ngw_ptr = (1 - p_lt30gt1) + p_lt30gt1 * p_ngw  # [batch_size, 1]
    true_gw_ptr = 1 - tf.math.exp(-tf.div_no_nan(pred_lt30_trans, gw_thres))  # [batch_size, 1]
    true_ngw_ptr = 1 - true_gw_ptr  # [batch_size, 1]
    preds_concat = tf.concat([preds_gw_ptr, preds_ngw_ptr], axis=1)  # [batch_size, 2]
    true_concat = tf.concat([true_gw_ptr, true_ngw_ptr], axis=1)  # [batch_size, 2]
    per_sample_loss = kl_divergence_loss(true_concat, preds_concat)
    mask = tf.reshape(is_valid_lt30, tf.shape(per_sample_loss))
    loss_gwd = tf.reduce_sum(per_sample_loss * mask)

    def generate_lt_by_level(pred_lt30):

        lt_1_5 = tf.cast(tf.where(pred_lt30 < 5.0, tf.ones_like(pred_lt30), tf.zeros_like(pred_lt30)), tf.float32)
        lt_5_10 = tf.cast(tf.where((pred_lt30 >= 5.0) & (pred_lt30 < 10.0), tf.ones_like(pred_lt30), tf.zeros_like(pred_lt30)), tf.float32)
        # lt_3_5 = tf.cast(tf.where((pred_lt30 >= 3.0) & (pred_lt30 < 5.0), tf.ones_like(pred_lt30), tf.zeros_like(pred_lt30)), tf.float32)
        # lt_5_10 = tf.cast(tf.where((pred_lt30 >= 5.0) & (pred_lt30 < 10.0), tf.ones_like(pred_lt30), tf.zeros_like(pred_lt30)), tf.float32)
        # lt_10_20 = tf.cast(tf.where((pred_lt30 >= 10.0) & (pred_lt30 < 20.0), tf.ones_like(pred_lt30), tf.zeros_like(pred_lt30)), tf.float32)
        lt_10_30 = tf.cast(tf.where(pred_lt30 >= 10.0, tf.ones_like(pred_lt30), tf.zeros_like(pred_lt30)), tf.float32)
        
        return (lt_1_5, lt_5_10, lt_10_30)
    
    # 拆分桶看
    lt_1_5, lt_5_10, lt_10_ = generate_lt_by_level(pred_lt30)
    
    eval_targets = [

        # ("rta_rtr_uninstall",       p_rtr,   retention_label, is_rta * uninstall_flag,       'auc'),
        # ("rta_rtr_install",         p_rtr,   retention_label, is_rta * install_flag,         'auc'),
        # ("half_rtb_rtr_uninstall",  p_rtr,   retention_label, is_half_rtb * uninstall_flag,  'auc'),
        # ("half_rtb_rtr_install",    p_rtr,   retention_label, is_half_rtb * install_flag,    'auc'),
        # ("rtb_rtr_uninstall",       p_rtr,   retention_label, is_rtb * uninstall_flag,       'auc'),
        # ("rtb_rtr_install",         p_rtr,   retention_label, is_rtb * install_flag,         'auc'),

        # ("rta_sumlt7_preds",      lt7_pred, valid_lt7, is_rta,      'linear_regression'),
        # ("half_rtb_sumlt7_preds", lt7_pred, valid_lt7, is_half_rtb, 'linear_regression'),
        # ("rtb_sumlt7_preds",      lt7_pred, valid_lt7, is_rtb,      'linear_regression'),

        ("rta_lt7_uninstall",      lt7_pred,    valid_lt7,    is_rta * uninstall_flag * is_valid_lt7,       'linear_regression'),
        ("rta_lt7_install",        lt7_pred,    valid_lt7,    is_rta * install_flag * is_valid_lt7,         'linear_regression'),
        ("half_rtb_lt7_uninstall", lt7_pred,    valid_lt7,    is_half_rtb * uninstall_flag * is_valid_lt7,  'linear_regression'),
        ("half_rtb_lt7_install",   lt7_pred,    valid_lt7,    is_half_rtb * install_flag * is_valid_lt7,    'linear_regression'),
        ("rtb_lt7_uninstall",      lt7_pred,    valid_lt7,    is_rtb * uninstall_flag * is_valid_lt7,       'linear_regression'),
        ("rtb_lt7_install",        lt7_pred,    valid_lt7,    is_rtb * install_flag * is_valid_lt7,         'linear_regression'),
        
        ('rta_lt30_install',             p_lt30,         pred_lt30,          is_rta * install_flag * is_valid_lt30,            'linear_regression'),
        ('rtb_lt30_install',             p_lt30,         pred_lt30,          is_rtb * install_flag * is_valid_lt30,            'linear_regression'),
        ('half_rtb_lt30_install',        p_lt30,         pred_lt30,          is_half_rtb * install_flag * is_valid_lt30,       'linear_regression'),
        ('rta_lt30_uninstall',           p_lt30,         pred_lt30,          is_rta * uninstall_flag * is_valid_lt30,          'linear_regression'),
        ('rtb_lt30_uninstall',           p_lt30,         pred_lt30,          is_rtb * uninstall_flag * is_valid_lt30,          'linear_regression'),
        ('half_rtb_lt30_uninstall',      p_lt30,         pred_lt30,          is_half_rtb * uninstall_flag * is_valid_lt30,     'linear_regression'),

        ('half_rtb_lt30_uninstall_gdt',            p_lt30,       pred_lt30,     is_half_rtb * uninstall_flag * is_gdt * is_valid_lt30,             'linear_regression'),
        ('half_rtb_lt30_uninstall_vivo',           p_lt30,       pred_lt30,     is_half_rtb * uninstall_flag * is_vivo * is_valid_lt30,            'linear_regression'),
        ('half_rtb_lt30_uninstall_oppo',           p_lt30,       pred_lt30,     is_half_rtb * uninstall_flag * is_oppo * is_valid_lt30,            'linear_regression'),
        ('half_rtb_lt30_uninstall_csj',            p_lt30,       pred_lt30,     is_half_rtb * uninstall_flag * is_csj * is_valid_lt30,             'linear_regression'),
        ('half_rtb_lt30_uninstall_qtt',            p_lt30,       pred_lt30,     is_half_rtb * uninstall_flag * is_qtt * is_valid_lt30,             'linear_regression'),
        
        ('half_rtb_lt30_install_gdt',            p_lt30,       pred_lt30,     is_half_rtb * install_flag * is_gdt * is_valid_lt30,             'linear_regression'),
        ('half_rtb_lt30_install_vivo',           p_lt30,       pred_lt30,     is_half_rtb * install_flag * is_vivo * is_valid_lt30,            'linear_regression'),
        ('half_rtb_lt30_install_oppo',           p_lt30,       pred_lt30,     is_half_rtb * install_flag * is_oppo * is_valid_lt30,            'linear_regression'),
        ('half_rtb_lt30_install_csj',            p_lt30,       pred_lt30,     is_half_rtb * install_flag * is_csj * is_valid_lt30,             'linear_regression'),
        ('half_rtb_lt30_install_qtt',            p_lt30,       pred_lt30,     is_half_rtb * install_flag * is_qtt * is_valid_lt30,             'linear_regression'),

        ('half_rtb_lt30_1_2',            p_lt30,       pred_lt30,     is_half_rtb * lt_1_5,        'linear_regression'),
        ('half_rtb_lt30_2_3',            p_lt30,       pred_lt30,     is_half_rtb * lt_5_10,        'linear_regression'),
        ('half_rtb_lt30_3_5',            p_lt30,       pred_lt30,     is_half_rtb * lt_10_,        'linear_regression'),

        ('half_rtb_lt30_uninstall_1_2',            p_lt30,       pred_lt30,     is_half_rtb * lt_1_5 * uninstall_flag,        'linear_regression'),
        ('half_rtb_lt30_uninstall_2_3',            p_lt30,       pred_lt30,     is_half_rtb * lt_5_10 * uninstall_flag,        'linear_regression'),
        ('half_rtb_lt30_uninstall_3_5',            p_lt30,       pred_lt30,     is_half_rtb * lt_10_ * uninstall_flag,        'linear_regression'),

        ('half_rtb_lt30_install_1_2',            p_lt30,       pred_lt30,     is_half_rtb * lt_1_5 * install_flag,        'linear_regression'),
        ('half_rtb_lt30_install_2_3',            p_lt30,       pred_lt30,     is_half_rtb * lt_5_10 * install_flag,        'linear_regression'),
        ('half_rtb_lt30_install_3_5',            p_lt30,       pred_lt30,     is_half_rtb * lt_10_ * install_flag,        'linear_regression'),
        ]
    
    lt_log_loss = tf.losses.log_loss((valid_lt7 - 1) / 6.0, (lt7_pred - 1) / 6.0, tf.cast(tf.ones_like(is_rta), tf.float32) * is_valid_lt7, reduction="weighted_sum")
    
    my_step = config.get_step()
    TOPIC_ID = config.get_extra_param('TOPIC_ID')
    TOPIC_ID_Mean = tf.reduce_mean(TOPIC_ID)
    total_dense_var = config.get_dense_trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in total_dense_var])
    log_weight = tf.cond(tf.greater(my_step, 100),lambda: tf.ones_like(log_normal_loss) * 1, lambda: tf.ones_like(log_normal_loss) * 1.0)
    
    mask1 = tf.cast(valid_retention_label > 0.5, tf.float32) # 如果是1，则mask1=1，否则0
    mask0 = 1.0 - mask1 # 如果是0，则mask0=1，否则0
    m = 0.05
    pairwise_loss = (mask1 * tf.maximum((1 - m) * fusion_prtr - p_rtr, 0) + mask0 * tf.maximum(p_rtr - (1 + m) * 0.5, 0)) * is_rtb
    pairwise_loss = tf.reduce_sum(pairwise_loss)

    tf.summary.histogram("rtr_loss", rtr_loss)
    tf.summary.histogram("lt7_loss", lt7_loss)
    tf.summary.histogram("lt_log_loss", lt_log_loss)
    tf.summary.histogram("log_normal_loss", log_normal_loss)
    tf.summary.histogram("loss_gwd", loss_gwd)
    tf.summary.histogram("lt30gt1_loss", lt30gt1_loss)

    print_label_op = tf.cond(
        tf.equal(tf.mod(my_step, 10), 1),
        lambda: tf.print("step:", my_step, 'TOPIC_ID_Mean:',TOPIC_ID_Mean, "rtr_loss:", rtr_loss, "lt7_loss:", lt7_loss, 'l2_loss', l2_loss, "lt_log_loss:", lt_log_loss,
                         "pairwise_loss:", pairwise_loss, "lt30gt1_loss:", lt30gt1_loss, "log_normal_loss:", log_normal_loss, "loss_gwd:", loss_gwd,
                         summarize=-1, output_stream=sys.stdout),
        lambda: tf.no_op())
    
    with tf.control_dependencies([print_label_op] + all_mixer_ops):
        loss = (lt7_loss * 2 + lt_log_loss * 0.3) * 1 + (log_normal_loss + loss_gwd) * log_weight + lt30gt1_loss
    
    # build model
    # config.set_feature_score_attr("is_retention", data_source_name="train")
    optimizers = []
    sparse_optimizer = config.optimizer.AdagradShareG2SumOptimizer(0.05, g2sum_decay_rate=1.0)
    sparse_optimizer.set_weight_clip(10)
    dense_optimizer = config.optimizer.MioDense(0.0002, ada_decay_rate=0.9999)
    sparse_optimizer.minimize(loss, var_list=[i for i in config.get_collection(config.GraphKeys.EMBEDDING_INPUT) if
                                              i not in CUSTOM_OPT_SETS])
    dense_optimizer.minimize(loss,
                             var_list=[i for i in config.get_collection(config.GraphKeys.TRAINABLE_VARIABLES) if
                                       i not in CUSTOM_OPT_SETS])
    optimizers.append(sparse_optimizer)
    optimizers.append(dense_optimizer)

    # 处理AssignAdd优化器
    for var in CUSTOM_OPT_SETS:
        opt = CUSTOM_OPT_SETS[var]
        if opt.get("opt_type", "") == "AssignAdd":
            custom_grad = opt.get("grad", None)
            assert custom_grad is not None
            optimizer = config.optimizer.AssignAddOptimizer(decay_rate=opt.get("decay_rate", 0), add_rate=opt.get("add_rate", 1))
            optimizer.minimize(loss, var_list=[var], custom_gradient={var.name: custom_grad})
            optimizers.append(optimizer)
    config.build_model(optimizer=optimizers, metrics=eval_targets)
    if UDP_OUTPUT_PRED == 1:
        log_dict = {
            'is_rta': is_rta,
            'is_rtb': is_rtb,
            'is_half_rtb': is_half_rtb,
            'install_flag': install_flag,
            'uninstall_flag': uninstall_flag,
            "offline_req_id_hash": config.get_extra_param('offline_req_id_hash', size=1, reversed=False, default_value=0.0),
            "creative_id": config.get_extra_param('creative_id', size=1, reversed=False, default_value=0.0),
            'retention_label': retention_label,
            'lt7': lt7,
            'pred_lt30': pred_lt30,
            'true_lt30': lt30,
            'rtr_pred': p_rtr,
            'lt7_pred': lt7_pred,
            'p_lt30_mu': p_lt30_mu,
            'p_lt30_sigma': p_lt30_sigma,
            'lt30_pred': p_lt30,
            'time_stamp': config.get_extra_param('time_stamp')
        }
        config.add_run_hook(DumpTensorHook('lt7_wangzihan', log_dict), 'custom_dump_tensor_hook')
else:
    targets = [
        ('rta2_rtr',     p_rtr),
        ('rta_rtr',      p_rtr),
        ('rtb_rtr',      p_rtr),
        ('rta2_lt7',     lt7_pred),
        ('rta_lt7',      lt7_pred),
        ('rtb_lt7',      lt7_pred),
        ('rta2_lt7gt2',  lt7_prob_list[0]),
        ('rta2_lt7gt3',  lt7_prob_list[1]),
        ('rta2_lt7gt4',  lt7_prob_list[2]),
        ('rta2_lt7gt5',  lt7_prob_list[3]),
        ('rta2_lt7gt6',  lt7_prob_list[4]),
        ('rta2_lt7gt7',  lt7_prob_list[5]),
        ('rta2_lt30',     tf.minimum(p_lt30, 30)),
        ('rta_lt30',      tf.minimum(p_lt30, 30)),
        ('rtb_lt30',      tf.minimum(p_lt30, 30)),

    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config(
        'infer_conf/models/reflux_model_fusion_v1/',
        targets, input_type=3,
        extra_preds=q_names
    )
