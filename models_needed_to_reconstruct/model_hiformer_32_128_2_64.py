from __future__ import print_function

import os
import argparse
import sys
from tensorflow.keras.backend import expand_dims, repeat_elements
from tensorflow.keras.backend import sum as backend_sum
import numpy as np
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

# get labels
all_labels = ['retention_label', 'lt7', 'app_usage_duration_0d_bin','app_usage_duration_1d_bin','app_usage_duration_2d_bin', 'valid_duration_lt7', 'is_not_0vv_lt7', 'valid_retention_label',
                         'app_usage_duration_3d_bin', 'app_usage_duration_4d_bin', 'app_usage_duration_5d_bin', 'app_usage_duration_6d_bin', 'app_usage_duration_7d_bin', 'valid_lt7', 'fusion_prtr']
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

def simple_dense_network_with_ppnet(inputs, ppnet_input, units, ppnet_unit, name, act=tf.nn.relu, dropout = 0.0, batch_norm = False, training=False):
    output = inputs
    for i, unit in enumerate(units):
        if i > 0:
            output = tf.multiply(simple_lhuc_network(ppnet_input, ppnet_unit, output.get_shape().as_list()[-1], 'ppnet_{}_{}'.format(name, i)), output)
        output = mio_dense_layer_with_bn(output, unit, act, batch_norm, name='dense_{}_{}'.format(name, i), training=training)
        output = tf.layers.dropout(output, rate=dropout*bool(args.mode == 'train'))
    return output

def simple_lhuc_network(inputs, unit1, unit2, name):
    output = mio_dense_layer(inputs, unit1, tf.nn.relu, name='dense_{}_relu'.format(name))
    output = 2.0 * mio_dense_layer(output, unit2, tf.nn.sigmoid, name='dense_{}_sigmoid'.format(name))
    return output

def single_head_hetero_attention(query_input, att_emb_size=3, scope='attention'):
    with tf.variable_scope(scope):
        D = query_input.get_shape().as_list()[-1]
        T = query_input.get_shape().as_list()[1]

        Q = tf.get_variable('q_trans_matrix', shape=(T, D, att_emb_size))
        K = tf.get_variable('k_trans_matrix', shape=(T, D, att_emb_size))
        W_logits = tf.get_variable("Matrices_Wi", shape=(T, D, D))
        W_logits_constraint = (W_logits + tf.transpose(W_logits, perm=[0, 2, 1])) * 0.5

        # token-specific 投影矩阵
        querys = tf.einsum('btd,tde->bte', query_input, Q)
        keys   = tf.einsum('btd,tde->bte', query_input, K)
        values = tf.einsum('bni, nio -> bno', query_input, W_logits_constraint)

        inner_product = tf.matmul(querys, keys, transpose_b=True) / tf.sqrt(float(att_emb_size))

        att_scores = tf.nn.softmax(inner_product)
        att_scores = (att_scores + tf.transpose(att_scores, perm=[0, 2, 1])) * 0.5

        result = tf.einsum('btj,bjd->btd', att_scores, values)
        output = tf.reshape(result, [-1, T * D])
        output = tf.reshape(output, [-1, T, D])
    return output

def multi_head_hetero_attention(X, nh=5, att_emb_size=6, scope='attention'):
    with tf.variable_scope(scope):
        D = X.get_shape().as_list()[-1]
        T = X.get_shape().as_list()[1]

        Q = tf.get_variable('q_trans_matrix', shape=(T, D, att_emb_size * nh))                          # [T, D, emb_size * nh]
        K = tf.get_variable('k_trans_matrix', shape=(T, D, att_emb_size * nh))                          # [T, D, emb_size * nh]
        V = tf.get_variable("v_trans_matrix", shape=(T, D, D * nh))                                     # [T, D, D * nh]
        O = tf.get_variable("o_trans_matrix", shape=(T, D * nh, D))                                     # [T, D * nh, D]

        # 2. token-specific 投影矩阵
        querys_0 = tf.einsum('btd,tde->bte', X, Q)                                                      # [bs, T, emb_size * nh]
        keys_0   = tf.einsum('btd,tde->bte', X, K)                                                      # [bs, T, emb_size * nh]
        values_0 = tf.einsum('bni, nio -> bno', X, V)                                                   # [bs, T, D * nh]

        ## 以下部分的 pattern 会被优化脚本自动识别并替换
        querys = tf.stack(tf.split(querys_0, nh, axis=2))  							                    # [nh, bs, T, emb_size]		
        keys = tf.stack(tf.split(keys_0, nh, axis=2))  											        # [nh, bs, T, emb_size]
        values = tf.stack(tf.split(values_0, nh, axis=2))  								                # [nh, bs, T, D]

        inner_product = tf.matmul(querys, keys, transpose_b=True) / tf.sqrt(float(att_emb_size))        # [nh, bs, T, T]
        att_scores = tf.nn.softmax(inner_product)                                                       # [nh, bs, T, T]
    
        result_1 = tf.matmul(att_scores, values)                                                        # [nh, bs, T, D]
        result_2 = tf.transpose(result_1, perm=[1, 2, 0, 3])                                            # [bs, T, nh, D]
        result = tf.reshape(result_2, (-1, T, nh * D))                                                  # [bs, T, nh * D]
        outputs = tf.einsum('bni, nio -> bno', result, O)	                                            # [bs, T, D]
    return outputs   

def multi_head_hiformer_low_rank(X, nh=5, rq=64, rk=64, rv=64, scope='attention'):
    """
    低秩多头注意力机制
    
    参数示例：
        - 输入 X: [B, N, K] = [2048, 256, 64] (RankMixer中N=256个chunk, K=64维度)
        - nh=4, rq=rk=rv=8
        - head_dim = K // nh = 64 // 4 = 16
    """
    with tf.variable_scope(scope):
        D = X.get_shape().as_list()[-1]           # D = K = 64 (特征维度)
        T = X.get_shape().as_list()[1]            # T = N = 256 (token/chunk数)
        batchsize = tf.shape(X)[0]                # B = 2048 (batch_size)
        head_dim = D // nh                        # 每个头的维度 = 64 // 4 = 16

        # ============ Step 1: Flatten输入 ============
        # 将 [B, T, D] reshape为 [B, 1, T*D]，便于矩阵乘法
        # [2048, 256, 64] → [2048, 1, 16384]
        X_flat = tf.reshape(X, (batchsize, 1, T * D))

        # ============ Step 2: Q/K/V 低秩分解 ============
        # 目标：用低秩矩阵分解避免存储巨大的 [T*D, T*D] 矩阵
        # 原始矩阵大小：16384 × 16384 ≈ 268M 参数
        # 低秩分解后：(16384 × 8) + (8 × 16384) ≈ 262K 参数 (减少约1000倍)
        
        # Q: 低秩分解 - 使用随机初始化（注意：这里没有预训练权重，低秩分解本身就是主要计算方式）
        # 使用 Xavier/He 初始化策略，考虑矩阵乘法的梯度稳定性
        # stddev = sqrt(2 / (fan_in + fan_out)) ≈ sqrt(2 / (T*D + T*D)) = sqrt(1/T*D)
        init_std = 0.001
        A_Q = tf.get_variable('Aq_trans_matrix', shape=(T * D, rq),
                              initializer=tf.truncated_normal_initializer(stddev=init_std))
        B_Q = tf.get_variable('Bq_trans_matrix', shape=(rq, T * D),
                              initializer=tf.truncated_normal_initializer(stddev=init_std))
        querys_0 = tf.matmul(tf.matmul(X_flat, A_Q), B_Q)  # [B, 1, T*D]

        # K: 同理
        init_std_k = 0.001
        A_K = tf.get_variable('Ak_trans_matrix', shape=(T * D, rk),
                              initializer=tf.truncated_normal_initializer(stddev=init_std_k))
        B_K = tf.get_variable('Bk_trans_matrix', shape=(rk, T * D),
                              initializer=tf.truncated_normal_initializer(stddev=init_std_k))
        keys_0 = tf.matmul(tf.matmul(X_flat, A_K), B_K)

        # V: 同理
        init_std_v = 0.001
        A_V = tf.get_variable('Av_trans_matrix', shape=(T * D, rv),
                              initializer=tf.truncated_normal_initializer(stddev=init_std_v))
        B_V = tf.get_variable('Bv_trans_matrix', shape=(rv, T * D),
                              initializer=tf.truncated_normal_initializer(stddev=init_std_v))
        values_0 = tf.matmul(tf.matmul(X_flat, A_V), B_V)

        # ============ Step 3: 多头拆分 ============
        # 将 [B, 1, T*D] reshape为 [B, T, nh, head_dim]
        # 然后 transpose 为 [nh, B, T, head_dim] 以便并行计算
        # [2048, 1, 16384] → [2048, 256, 4, 16] → [4, 2048, 256, 16]
        
        querys = tf.reshape(querys_0, [batchsize, T, nh, head_dim])   # [2048, 256, 4, 16]
        querys = tf.transpose(querys, [2, 0, 1, 3])                   # [4, 2048, 256, 16]

        keys = tf.reshape(keys_0, [batchsize, T, nh, head_dim])       # [2048, 256, 4, 16]
        keys = tf.transpose(keys, [2, 0, 1, 3])                       # [4, 2048, 256, 16]

        values = tf.reshape(values_0, [batchsize, T, nh, head_dim])   # [2048, 256, 4, 16]
        values = tf.transpose(values, [2, 0, 1, 3])                   # [4, 2048, 256, 16]

        # ============ Step 4: Attention 计算 ============
        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        # [4, 2048, 256, 16] @ [4, 2048, 16, 256] = [4, 2048, 256, 256]
        scores = tf.matmul(querys, keys, transpose_b=True)
        scores = scores / tf.sqrt(float(head_dim))
        attn_weights = tf.nn.softmax(scores)                          # [4, 2048, 256, 256]

        # 加权求和: Attention @ V
        # [4, 2048, 256, 256] @ [4, 2048, 256, 16] = [4, 2048, 256, 16]
        attn_output = tf.matmul(attn_weights, values)

        # ============ Step 5: 合并多头结果 ============
        # [4, 2048, 256, 16] → [2048, 256, 4, 16] → [2048, 256, 64]
        attn_output = tf.transpose(attn_output, [1, 2, 0, 3])         # [2048, 256, 4, 16]
        attn_output = tf.reshape(attn_output, [batchsize, T, D])      # [2048, 256, 64]

        # ============ Step 6: 输出投影 ============
        # 对每个位置 T 做线性变换: [D] → [D]
        # O: [256, 64, 64], 每个位置有独立的投影矩阵
        O = tf.get_variable("o_trans_matrix", shape=(T, D, D))        # [256, 64, 64]
        # [B, T, D] @ [T, D, D] → [B, T, D]
        # einsum: bti,tio->bto 表示 batch维度b，位置维度t，输入维度i，输出维度o
        output = tf.einsum('bti,tio->bto', attn_output, O)            # [2048, 256, 64]

    return output

def loop_regression(input_embedding,
                     dnn_layers=[16],
                     cond_probs=7,
                     scope='default',
                     residual_logit=None):
    pred_single_day_duration_dict = {}
    pred = 0
    with tf.variable_scope("loop_regression_duration_{}".format(scope)):
        
        for i in range(cond_probs):
            _name = "{}".format(i)
            if i == 0:
                out = simple_dense_network(input_embedding, dnn_layers, name="single_day_duration_{}".format(_name))
            else:
                out = simple_dense_network(tf.concat([input_embedding, previous_outputs], axis=-1), dnn_layers, name="single_day_duration_{}".format(_name))
            previous_outputs = out
            out = mio_dense_layer(out, 1, activation=None, name="single_day_duration_{}_last_fc".format(_name))
            if residual_logit is not None:
                pred_single_day_duration_dict[_name] = tf.nn.sigmoid(out+residual_logit[i])
            else:
                pred_single_day_duration_dict[_name] = tf.nn.sigmoid(out)
        for i in range(cond_probs):
            _name = "{}".format(i)
            pred += pred_single_day_duration_dict[_name]
        
    return {
        "single_day_duration": pred_single_day_duration_dict,
        "pred": pred
    }

def order_regression_with_ppnet(input_embedding,
                     ppnet_input,
                     dnn_layers=[16],
                     ppnet_unit=16,
                     cond_probs=6,
                     scope='default',
                     residual_logit=None,
                     training=False):
    pred_cond_probs_dict = {}
    pred_cut_probs_dict = {}
    pred = 1
    with tf.variable_scope("order_regression_{}".format(scope)):
        for i in range(cond_probs):
            _name = "{}_{}".format(i + 2, i + 1)
            out = simple_dense_network_with_ppnet(input_embedding, ppnet_input, dnn_layers, ppnet_unit, name="cond_prob_{}".format(_name), act=tf.nn.relu, dropout=0.0, batch_norm=False, training=training)
            out = mio_dense_layer(out, 1, activation=None, name="cond_prob_{}_last_fc".format(_name))
            if residual_logit is not None:
                pred_cond_probs_dict[_name] = tf.nn.sigmoid(out+residual_logit[i+1])
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
    return {
        "cond_prob": pred_cond_probs_dict,
        "cut_prob": pred_cut_probs_dict,
        "pred": pred
    }

def shared_bottom_rtr_tower(input_embedding, ppnet_input, gating_input, tower_units=[64, 32], ppnet_unit=64, num_tasks=5, scope='shared_bottom_rtr_tower', training=False):

    pred_probs_dict = {}
    
    with tf.variable_scope(scope):

        shared_representation = simple_dense_network_with_ppnet(
            input_embedding, ppnet_input, tower_units, ppnet_unit,
            name="shared_backbone", act=tf.nn.relu, dropout=0.0,
            batch_norm=False, training=training
        )
        task_bias_hidden = simple_dense_network(gating_input, [32, 16], 'gating_bias_tower_hidden', act=tf.nn.relu, dropout = 0.0)

        for i in range(num_tasks):
            with tf.variable_scope(f"task_head_{i}"):
                task_bias = mio_dense_layer(task_bias_hidden, units=1, activation=None, name=f"gating_bias_tower_{i}")
                task_logit = mio_dense_layer(shared_representation, units=1, activation=None, name=f"task_logit_{i}")
                
                fused_logit = task_logit + task_bias
                pred_probs_dict[i] = tf.nn.sigmoid(fused_logit, name=f"task_prob_{i}")

        pred_probs_concat = tf.reshape(tf.concat([pred_probs_dict[i] for i in range(num_tasks)], axis=1), (-1, num_tasks))

    return {
        "pred_probs_dict": pred_probs_dict,
        "pred_probs_concat": pred_probs_concat,
    }

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

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * tf.pow(x,3))))

def ffn(hidden_states,
        d_model,
        d_ff,
        scope="ffn",
        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        reshaped_input = tf.squeeze(hidden_states, axis=1)
        hidden_states = mio_dense_layer(reshaped_input, d_ff, activation=None, name='dense_1')
        hidden_states = gelu(hidden_states)
        hidden_states = mio_dense_layer(hidden_states, d_model, activation=None, name='dense_2')
        output_states = tf.expand_dims(hidden_states, axis=1)
        return output_states

def RankMixer_simplified(hidden_states, d_model, d_ff, scope="RankMixer", is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        T = hidden_states.get_shape().as_list()[1]  # l = T (token数), 例如 16
        
        #--------------------------Mixing Start----------------------------
        # multi_head_hiformer_low_rank 在 T 维度上做attention：
        #   - 输入：[B, T, D] = [2048, 16, 1024]
        #   - 内部执行流程：
        #     1) Q/K/V低秩分解
        #     2) reshape分头
        #     3) attention计算
        #     4) 合并多头结果
        #     5) O投影
        # 从配置中读取注意力超参数
        
        mixing_h_hmm = multi_head_hiformer_low_rank(
            hidden_states, 
            nh=mixer_config.attention_nh, 
            rq=mixer_config.attention_rq, 
            rk=mixer_config.attention_rk, 
            rv=mixer_config.attention_rv, 
            scope='attention'
        )
        # mixing_h_hmm: [B, T, D] = [2048, 16, 1024]
        
        mixing_h = mixing_h_hmm
        #--------------------------Mixing End----------------------------

        # 残差连接 + RMS Norm
        # hidden_states: [B, T, D] + mixing_h: [B, T, D] → mixed_hidden_states: [B, T, D]
        mixed_hidden_states = rms_norm(hidden_states + mixing_h, scope="post_mixing_norm")
        # mixed_hidden_states: [B, T, D] = [2048, 16, 1024]
        
        # ==================== Token-wise FFN 阶段 ====================
        # 目标：每个token独立过FFN，学习token特定的非线性变换
        all_token_outputs = []
        for i in range(T):  # 遍历 T=16 个token
            # 取出第i个token: [B, 1, D] 例如 [2048, 1, 1024]
            token_slice = tf.slice(mixed_hidden_states, [0, i, 0], [-1, 1, -1])
            
            with tf.variable_scope(f"Token_{i}_FFN", reuse=reuse):
                # ffn内部流程：
                #   - squeeze: [B, 1, D] → [B, D] = [2048, 1024]
                #   - Linear1: [B, D] @ [D, d_ff] → [B, d_ff] = [2048, 2048] (d_ff=2*D)
                #   - GELU激活
                #   - Linear2: [B, d_ff] @ [d_ff, D] → [B, D] = [2048, 1024]
                #   - expand_dims: [B, D] → [B, 1, D] = [2048, 1, 1024]
                token_output = ffn(token_slice, d_model, d_ff)
            # token_output: [B, 1, D] = [2048, 1, 1024]
            all_token_outputs.append(token_output)

        # 合并所有token的FFN输出
        ffn_output = tf.concat(all_token_outputs, axis=1)  # [B, T, D] = [2048, 16, 1024]
        
        # 残差连接 + RMS Norm
        # mixed_hidden_states: [B, T, D] + ffn_output: [B, T, D] → final_hidden_states: [B, T, D]
        final_hidden_states = rms_norm(mixed_hidden_states + ffn_output, scope="post_ffn_norm")
        # final_hidden_states: [B, T, D] = [2048, 16, 1024]

        return final_hidden_states, mixing_h_hmm
    
class TensorsPrintHook(config.training.RunHookBase):
    def __init__(self, your_custom_tensor_dict):
        self.custom_tensor_dict = your_custom_tensor_dict

    def before_step_run(self, step_run_context):
        return config.training.StepRunArgs(fetches=self.custom_tensor_dict)

    def after_step_run(self, step_run_context, step_run_values):
        result = step_run_values.result
        output = ""
        for name, val in result.items():
            output += "{}: {},".format(name, val)
        print(output)
need_print_tensor_dict = {}

class MixerConfig:
    def __init__(self):
        self.token_num = 32
        self.d_kv = 128 # origin = 512 每个head的维度，head数量等于token数量
        self.num_block = 2 #层数
        self.d_ff_coefficient = 2 # ffn中间层维度倍数，默认2倍
        self.expert_num = 1 #专家数量
        self.k = 1 #TopK
        self.alpha = 0.0 #负载均衡损失函数权重
        
        # ==================== 注意力相关超参数 ====================
        # 低秩多头注意力 (multi_head_hiformer_low_rank) 配置
        self.attention_nh = 4  # 多头注意力的头数，默认4
        self.attention_rq = 64  # Q矩阵低秩分解的秩 (reduced dim for Query)
        self.attention_rk = 64  # K矩阵低秩分解的秩 (reduced dim for Key)
        self.attention_rv = 64  # V矩阵低秩分解的秩 (reduced dim for Value)
        # 注意: head_dim = K // nh = (d_kv // token_num) // attention_nh

mixer_config = MixerConfig()
is_training = (args.mode == 'train')

# global variables
ops = []
# id features
input_emb_dim = 4
wide_emb_dim = 4
request_feature_slots = [56, 58]
tense_slots = [61, 62, 63, 64, 65, 66, 161, 162, 163]
user_common_slots = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 83, 84, 86, 87, 88, 91, 92, 93, 94, 95, 96, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115] + \
                [255,256,258,259,260,270,271,272,273,274,275,
                276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 
                303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 
                328, 329, 330, 331, 332, 333, 334, 335,336,337,338,339,340,341,
                342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
                116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,143,144,145]
rtb_feature_slots = [53, 54, 55, 57, 100, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                     210, 211, 212, 213, 214, 215, 216, 217, 218,219,220, 221, 222, 223, 224, 225, 226, 227, 228,
                     229, 230, 231, 232, 233, 234, 235, 236,237,238,239, 240, 241, 242, 243, 244, 245, 246, 247,
                     248, 249, 250, 251, 252, 253]
rtb_feature_slots_item = [2, 3, 5, 60]
user_common_embs = config.new_embedding("user_common_input", dim=input_emb_dim, slots=user_common_slots,**get_predict_param())
request_feature_embs = config.new_embedding("request_feature_embs", dim=input_emb_dim, slots=request_feature_slots,**get_predict_param())
day_of_hour_bias_embedding = config.new_embedding("day_of_hour_bias_emb", dim=8, slots=[254])
tense_embs = config.new_embedding("tense_embs", dim=input_emb_dim, slots=tense_slots,**get_predict_param())
tense_embs = tf.concat([tense_embs, day_of_hour_bias_embedding], axis=-1)
active_app_emb = config.new_embedding("active_app_emb", dim=16, slots=[150,151,82,85,164,165], **get_predict_param())
install_app_emb = config.new_embedding("install_app_emb", dim=16, slots=[152,153,154,155,156,157,158,159], **get_predict_param())
channel_mix = config.new_embedding("channel_mix", dim=4, slots=[13, 90, 160])
usergroup_mix_embedding =  config.get_extra_param("usergroup_mix", size=4, **get_predict_param())
colossus_feature_embedding =  config.get_extra_param("colossus_feature", size=12, **get_predict_param())

# only rtb
rtb_feature = config.new_embedding("rtb_feature", dim=input_emb_dim, slots=rtb_feature_slots,**get_predict_param())
rate_mix_embedding =  config.get_extra_param("rate_mix", size=54, **get_predict_param())
mlp_ppnet_input_slots = [50, 51, 52, 59, 89]
mlp_ppnet_input_embs = config.new_embedding("mlp_ppnet_input_embs", dim=input_emb_dim, slots=mlp_ppnet_input_slots, **get_predict_param())

promotion_id_emb = config.new_embedding("promotion_id_emb", dim=input_emb_dim, slots=[7])
channel_type_emb = config.new_embedding("channel_type_emb", dim=input_emb_dim, slots=[8])
rtb_feature_item = config.new_embedding("rtb_feature_item", dim=input_emb_dim, slots=rtb_feature_slots_item)

# 底层输入特征的门控输入特征
# emb_ppnet_input = tf.stop_gradient(tf.concat([tense_embs], axis=1))
# 顶层次留存模型的门控输入特征
mlp_ppnet_input = tf.stop_gradient(tf.concat([request_feature_embs, channel_mix, channel_type_emb, mlp_ppnet_input_embs], axis=1))

min15_of_day_bias_embedding = config.new_embedding("min15_of_day_bias_emb", dim=8, slots=[1000],**get_predict_param())
holiday_type_embedding = config.new_embedding("holiday_type_emb", dim=8, slots=[1002],**get_predict_param())
cross_features = tf.math.multiply(holiday_type_embedding, day_of_hour_bias_embedding, name='explicit_cross_features')
ple_gating_input = tf.concat([holiday_type_embedding, day_of_hour_bias_embedding, min15_of_day_bias_embedding, cross_features], axis=1)

feature_list = [
    user_common_embs, request_feature_embs, tense_embs, active_app_emb, 
    install_app_emb, channel_mix, usergroup_mix_embedding, 
    colossus_feature_embedding, rate_mix_embedding, rtb_feature, 
    mlp_ppnet_input_embs, promotion_id_emb, channel_type_emb, 
    rtb_feature_item, day_of_hour_bias_embedding, min15_of_day_bias_embedding, holiday_type_embedding
]
final_concatenated_features = tf.concat(feature_list, axis=1) # [2048 1326]
total_input_dim = final_concatenated_features.get_shape()[-1] # 1326
remainder = total_input_dim % mixer_config.token_num
if remainder != 0:
    padding_size = mixer_config.token_num - remainder
    batch_size = tf.shape(final_concatenated_features)[0] # 2048
    padding_tensor = tf.zeros([batch_size, padding_size], dtype=final_concatenated_features.dtype)
    padded_features = tf.concat([final_concatenated_features, padding_tensor], axis=1)
else:
    padded_features = final_concatenated_features

total_padded_dim = padded_features.get_shape().as_list()[-1]
chunk_dim_per_token = total_padded_dim // mixer_config.token_num
reshaped_features = tf.reshape(
    padded_features, 
    [-1, mixer_config.token_num * chunk_dim_per_token]
)

token_slices_3d = tf.split(
    reshaped_features, 
    num_or_size_splits=mixer_config.token_num, 
    axis=1
)  # [token_num, bs, token_dim]
all_processed_tokens = []
with tf.variable_scope("shared_token_projection_scope", reuse=tf.AUTO_REUSE):
    for token_slice_2d in token_slices_3d:
        processed_slice = mio_dense_layer(
            token_slice_2d,   # [bs, token_dim]
            mixer_config.d_kv,
            activation=None,
            name="shared_token_projection"
        )
        all_processed_tokens.append(processed_slice)

hidden_states = tf.stack(all_processed_tokens, axis=1) # [2048   16 1024]

with tf.variable_scope("RankMixer_backbone"):
    mixing_ffn_dim = mixer_config.d_kv * mixer_config.d_ff_coefficient
    expert_num = mixer_config.expert_num
    mixer_regloss = 0

    hmm1 = hidden_states

    for i in range(mixer_config.num_block):
        hidden_states, mixing_h_hmm = RankMixer_simplified(
            hidden_states,
            mixer_config.d_kv,
            mixing_ffn_dim,
            scope=f"RankMixer_block_{i}",
            is_training=is_training
        )
        # mixer_regloss += mixer_config.alpha * load_balance_loss
    
    rankmixer_output = tf.reduce_mean(hidden_states, axis=1, name="rankmixer_mean_pooling_output")

with tf.name_scope('model'):

    task_names = ['rtr', 'lt7', 'duration']
    tower_units = [64, 32]

    gating_main_feature = mio_dense_layer(rankmixer_output, units=32, activation=tf.nn.relu, name='gating_main_feature')
    pred_probs = shared_bottom_rtr_tower(input_embedding=rankmixer_output, ppnet_input=mlp_ppnet_input, 
                                         gating_input=tf.concat([ple_gating_input, gating_main_feature], axis=1),tower_units=tower_units, 
                                         ppnet_unit=64, num_tasks=4, scope='rtr_tower', training=training)
    pred_probs_dict = pred_probs['pred_probs_dict']
    pred_probs_concat = pred_probs['pred_probs_concat']
    
    duration_preds = loop_regression(rankmixer_output, dnn_layers=tower_units, scope='duration')
    single_day_duration_pred_list = [duration_preds['single_day_duration']['{}'.format(i)] for i in range(7)]
    duration_pred = duration_preds['pred']

    lt7_preds = order_regression_with_ppnet(rankmixer_output, mlp_ppnet_input, tower_units, 64, 6, scope='regression_tower', residual_logit=single_day_duration_pred_list, training=training)
    lt7_cond_prob_list = [lt7_preds['cond_prob']['{}_{}'.format(i + 2, i + 1)] for i in range(6)]
    lt7_prob_list = [lt7_preds['cut_prob']['{}'.format(i + 2)] for i in range(6)]
    lt7_pred = lt7_preds['pred']
    
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
    valid_retention_label = get_label('valid_retention_label')
    fusion_prtr = get_label('fusion_prtr')
    valid_duration_lt7 = get_label('valid_duration_lt7')
    is_not_0vv_lt7 = get_label('is_not_0vv_lt7')
    retention_label = clip_to_boolean(get_label('retention_label'))

    holiday_type = tf.cast(config.get_extra_param('holiday_type', size=1, reversed=False, default_value=0.0), tf.int32)
    holiday_type_onehot = tf.reshape(tf.one_hot(holiday_type, depth=4), (-1, 4))
    p_rtr = tf.reshape(tf.reduce_sum(holiday_type_onehot * pred_probs_concat, axis=1), (-1, 1))

    # holiday_type = 0: 工作日 -> 工作日
    # holiday_type = 1: 工作日 -> 周末
    # holiday_type = 2: 周末/节假日 -> 工作日
    # holiday_type = 3: 周末/节假日 -> 周末（会有节假日 -> 周末的数据吗？）
    # holiday_type = 4: 任意 -> 节假日
    is_holiday_type_0 = config.get_extra_param('is_holiday_type_0', size=1, reversed=False, default_value=0.0)
    is_holiday_type_1 = config.get_extra_param('is_holiday_type_1', size=1, reversed=False, default_value=0.0)
    is_holiday_type_2 = config.get_extra_param('is_holiday_type_2', size=1, reversed=False, default_value=0.0)
    is_holiday_type_3 = config.get_extra_param('is_holiday_type_3', size=1, reversed=False, default_value=0.0)
    # is_holiday_type_4 = config.get_extra_param('is_holiday_type_4', size=1, reversed=False, default_value=0.0)

    is_day_1 = config.get_extra_param('is_day_1', size=1, reversed=False, default_value=0.0)
    is_day_2 = config.get_extra_param('is_day_2', size=1, reversed=False, default_value=0.0)
    is_day_3 = config.get_extra_param('is_day_3', size=1, reversed=False, default_value=0.0)
    is_day_4 = config.get_extra_param('is_day_4', size=1, reversed=False, default_value=0.0)
    is_day_5 = config.get_extra_param('is_day_5', size=1, reversed=False, default_value=0.0)
    is_day_6 = config.get_extra_param('is_day_6', size=1, reversed=False, default_value=0.0)
    is_day_7 = config.get_extra_param('is_day_7', size=1, reversed=False, default_value=0.0)

    # avg_expert_gate = tf.reduce_mean(gate, axis=[0, 1])
    # gate_tensors_to_monitor = {}
    # gate_tensors_to_monitor[f"mixer_gate_L{i}_avg"] = avg_expert_gate

    
    duration_label = []
    for day_num in range(8):
        param_name =  f'app_usage_duration_{day_num}d_bin'  # 参数名格式如 'app_usage_duration_0d_bin'
        param_value = get_label(param_name)
        duration_label.append(param_value)
        
    rtr_targets = [
        ('rtr_holiday_type_0',     pred_probs_dict[0],    retention_label,     is_holiday_type_0,     'auc'),
        ('rtr_holiday_type_1',     pred_probs_dict[1],    retention_label,     is_holiday_type_1,     'auc'),
        ('rtr_holiday_type_2',     pred_probs_dict[2],    retention_label,     is_holiday_type_2,     'auc'),
        ('rtr_holiday_type_3',     pred_probs_dict[3],    retention_label,     is_holiday_type_3,     'auc'),
        # ('rtr_holiday_type_4',     pred_probs_dict[4],    retention_label,     is_holiday_type_4,     'auc')
    ]
    q_name, preds, labels, weights, auc = zip(*rtr_targets)
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
                           lt7_cond_weight_list[i] * is_rta, "auc"))
        lt7_target.append(("rtb_lt7_cond_{}_{}".format(i+2, i+1), lt7_cond_prob_list[i], lt7_cut_list[i],
                           lt7_cond_weight_list[i] * is_rtb, "auc"))
        lt7_target.append(("half_rtb_lt7_cond_{}_{}".format(i+2, i+1), lt7_cond_prob_list[i], lt7_cut_list[i],
                           lt7_cond_weight_list[i] * is_half_rtb, "auc"))
    lt7_prob_targets = []
    for i in range(6):
        lt7_prob_targets.append(("rta_lt7_leq_{}".format(i+2), lt7_prob_list[i], lt7_cut_list[i], is_rta, "auc"))
        lt7_prob_targets.append(("rtb_lt7_leq_{}".format(i+2), lt7_prob_list[i], lt7_cut_list[i], is_rtb, "auc"))
        lt7_prob_targets.append(("half_rtb_lt7_leq_{}".format(i+2), lt7_prob_list[i], lt7_cut_list[i], is_half_rtb, "auc"))

    lt7_target_for_loss = []
    for i in range(6):
        lt7_target_for_loss.append(("lt7_cond_{}_{}".format(i+2, i+1), lt7_cond_prob_list[i], lt7_cut_list[i],
                           lt7_cond_weight_list[i] * tf.cast(tf.ones_like(is_rta), tf.float32), "auc"))
    q_name, preds, labels, weights, auc = zip(*lt7_target_for_loss)
    lt7_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    
    duration_targets = []
    for i in range(7):
        duration_targets.append(("rta_single_day_duration_{}".format(i), single_day_duration_pred_list[i], duration_label[i],
                           is_rta, "linear_regression"))
        duration_targets.append(("rtb_single_day_duration_{}".format(i), single_day_duration_pred_list[i], duration_label[i],
                           is_rtb, "linear_regression"))
        duration_targets.append(("half_rtb_single_day_duration_{}".format(i), single_day_duration_pred_list[i], duration_label[i],
                           is_half_rtb, "linear_regression"))
    
    q_name, preds, labels, weights, auc = zip(*duration_targets)
    duration_loss = tf.losses.log_loss(labels, preds, weights, reduction="weighted_sum")
    
    gdt_flag = config.get_extra_param('gdt_flag', size=1, reversed=False, default_value=0.0)
    vivo_flag = config.get_extra_param('vivo_flag', size=1, reversed=False, default_value=0.0)
    eval_targets = rtr_targets + [
        ('rta_rtr',          p_rtr,    retention_label,     is_rta,       'auc'),
        ('half_rtb_rtr',     p_rtr,    retention_label,     is_half_rtb,  'auc'),
        ('rtb_rtr',          p_rtr,    retention_label,     is_rtb,       'auc'),

        ("rta_sumlt7_preds",      lt7_pred, valid_lt7, is_rta,      'linear_regression'),
        ("half_rtb_sumlt7_preds", lt7_pred, valid_lt7, is_half_rtb, 'linear_regression'),
        ("rtb_sumlt7_preds",      lt7_pred, valid_lt7, is_rtb,      'linear_regression'),

        ("rta_rtr_uninstall",       p_rtr,   retention_label, is_rta * uninstall_flag,       'auc'),
        ("rta_rtr_install",         p_rtr,   retention_label, is_rta * install_flag,         'auc'),
        ("half_rtb_rtr_uninstall",  p_rtr,   retention_label, is_half_rtb * uninstall_flag,  'auc'),
        ("half_rtb_rtr_install",    p_rtr,   retention_label, is_half_rtb * install_flag,    'auc'),
        ("rtb_rtr_uninstall",       p_rtr,   retention_label, is_rtb * uninstall_flag,       'auc'),
        ("rtb_rtr_install",         p_rtr,   retention_label, is_rtb * install_flag,         'auc'),

        ("gdt_half_rtb_rtr_uninstall",  p_rtr,   retention_label, is_half_rtb * uninstall_flag * gdt_flag,  'auc'),
        ("gdt_half_rtb_rtr_install",  p_rtr,   retention_label, is_half_rtb * install_flag * gdt_flag,  'auc'),
        ("vivo_half_rtb_rtr_install",    p_rtr,   retention_label, is_half_rtb * install_flag * vivo_flag,    'auc'),
        ("vivo_half_rtb_rtr_uninstall",    p_rtr,   retention_label, is_half_rtb * uninstall_flag * vivo_flag,    'auc'),
        
        # ("rta_lt7_uninstall",      lt7_pred,    valid_lt7,    is_rta * uninstall_flag,       'linear_regression'),
        # ("rta_lt7_install",        lt7_pred,    valid_lt7,    is_rta * install_flag,         'linear_regression'),
        # ("half_rtb_lt7_uninstall", lt7_pred,    valid_lt7,    is_half_rtb * uninstall_flag,  'linear_regression'),
        # ("half_rtb_lt7_install",   lt7_pred,    valid_lt7,    is_half_rtb * install_flag,    'linear_regression'),
        # ("rtb_lt7_uninstall",      lt7_pred,    valid_lt7,    is_rtb * uninstall_flag,       'linear_regression'),
        # ("rtb_lt7_install",        lt7_pred,    valid_lt7,    is_rtb * install_flag,         'linear_regression'),
        
        # ("rta_lt7_0vv", lt7_pred, valid_lt7, is_rta * is_0vv_flag, 'linear_regression'),
        # ("rta_lt7_not_0vv", lt7_pred, valid_lt7, is_rta * is_not_0vv_flag, 'linear_regression'),
        # ("rtb_lt7_0vv", lt7_pred, valid_lt7, is_rtb * is_0vv_flag, 'linear_regression'),
        # ("rtb_lt7_not_0vv", lt7_pred, valid_lt7, is_rtb * is_not_0vv_flag, 'linear_regression'),
        # ("half_rtb_lt7_0vv", lt7_pred, valid_lt7, is_half_rtb * is_0vv_flag, 'linear_regression'),
        # ("half_rtb_lt7_not_0vv", lt7_pred, valid_lt7, is_half_rtb * is_not_0vv_flag, 'linear_regression'),
        
        # ("rta_valid_duration_lt7_uninstall", lt7_pred, valid_duration_lt7, is_rta * uninstall_flag, 'linear_regression'),
        # ("rta_valid_duration_lt7_install", lt7_pred, valid_duration_lt7, is_rta * install_flag, 'linear_regression'),
        # ("rtb_valid_duration_lt7_uninstall", lt7_pred, valid_duration_lt7, is_rtb * uninstall_flag, 'linear_regression'),
        # ("rtb_valid_duration_lt7_install", lt7_pred, valid_duration_lt7, is_rtb * install_flag, 'linear_regression'),
        # ("half_rtb_valid_duration_lt7_uninstall", lt7_pred, valid_duration_lt7, is_half_rtb * uninstall_flag, 'linear_regression'),
        # ("half_rtb_valid_duration_lt7_install", lt7_pred, valid_duration_lt7, is_half_rtb * install_flag, 'linear_regression'),

        ('rta_rtr_holiday_type_0',     p_rtr,    retention_label,     is_holiday_type_0 * is_rta,     'auc'),
        ('rta_rtr_holiday_type_1',     p_rtr,    retention_label,     is_holiday_type_1 * is_rta,     'auc'),
        ('rta_rtr_holiday_type_2',     p_rtr,    retention_label,     is_holiday_type_2 * is_rta,     'auc'),
        ('rta_rtr_holiday_type_3',     p_rtr,    retention_label,     is_holiday_type_3 * is_rta,     'auc'),
        # ('rta_rtr_holiday_type_4',     p_rtr,    retention_label,     is_holiday_type_4 * is_rta,     'auc'),

        ('rtb_rtr_holiday_type_0',     p_rtr,    retention_label,     is_holiday_type_0 * is_rtb,     'auc'),
        ('rtb_rtr_holiday_type_1',     p_rtr,    retention_label,     is_holiday_type_1 * is_rtb,     'auc'),
        ('rtb_rtr_holiday_type_2',     p_rtr,    retention_label,     is_holiday_type_2 * is_rtb,     'auc'),
        ('rtb_rtr_holiday_type_3',     p_rtr,    retention_label,     is_holiday_type_3 * is_rtb,     'auc'),
        # ('rtb_rtr_holiday_type_4',     p_rtr,    retention_label,     is_holiday_type_4 * is_rtb,     'auc'),

        ('half_rtb_rtr_holiday_type_0',     p_rtr,    retention_label,     is_holiday_type_0 * is_half_rtb,     'auc'),
        ('half_rtb_rtr_holiday_type_1',     p_rtr,    retention_label,     is_holiday_type_1 * is_half_rtb,     'auc'),
        ('half_rtb_rtr_holiday_type_2',     p_rtr,    retention_label,     is_holiday_type_2 * is_half_rtb,     'auc'),
        ('half_rtb_rtr_holiday_type_3',     p_rtr,    retention_label,     is_holiday_type_3 * is_half_rtb,     'auc'),
        # ('half_rtb_rtr_holiday_type_4',     p_rtr,    retention_label,     is_holiday_type_4 * is_half_rtb,     'auc'),

        ('half_rtb_rtr_d1',     p_rtr,    retention_label,     is_half_rtb * is_day_1,  'auc'),
        ('half_rtb_rtr_d2',     p_rtr,    retention_label,     is_half_rtb * is_day_2,  'auc'),
        ('half_rtb_rtr_d3',     p_rtr,    retention_label,     is_half_rtb * is_day_3,  'auc'),
        ('half_rtb_rtr_d4',     p_rtr,    retention_label,     is_half_rtb * is_day_4,  'auc'),
        ('half_rtb_rtr_d5',     p_rtr,    retention_label,     is_half_rtb * is_day_5,  'auc'),
        ('half_rtb_rtr_d6',     p_rtr,    retention_label,     is_half_rtb * is_day_6,  'auc'),
        ('half_rtb_rtr_d7',     p_rtr,    retention_label,     is_half_rtb * is_day_7,  'auc'),
        ]
    
    is_eval = int(os.getenv('IS_EVAL', 0))
    if is_eval:
        is_ios = config.get_extra_param('is_ios', size=1, reversed=False, default_value=0.0)
        is_android = config.get_extra_param('is_android', size=1, reversed=False, default_value=0.0)
        eval_targets += [
            ('ios_half_rtb_sumlt7_d1',     lt7_pred, valid_lt7,     is_half_rtb * is_day_1 * is_ios,  'linear_regression'),
            ('ios_half_rtb_sumlt7_d2',     lt7_pred, valid_lt7,     is_half_rtb * is_day_2 * is_ios,  'linear_regression'),
            ('ios_half_rtb_sumlt7_d3',     lt7_pred, valid_lt7,     is_half_rtb * is_day_3 * is_ios,  'linear_regression'),
            ('ios_half_rtb_sumlt7_d4',     lt7_pred, valid_lt7,     is_half_rtb * is_day_4 * is_ios,  'linear_regression'),
            ('ios_half_rtb_sumlt7_d5',     lt7_pred, valid_lt7,     is_half_rtb * is_day_5 * is_ios,  'linear_regression'),
            ('ios_half_rtb_sumlt7_d6',     lt7_pred, valid_lt7,     is_half_rtb * is_day_6 * is_ios,  'linear_regression'),
            ('ios_half_rtb_sumlt7_d7',     lt7_pred, valid_lt7,     is_half_rtb * is_day_7 * is_ios,  'linear_regression'),

            ('ios_half_rtb_rtr_d1',     p_rtr,    retention_label,     is_half_rtb * is_day_1 * is_ios,  'auc'),
            ('ios_half_rtb_rtr_d2',     p_rtr,    retention_label,     is_half_rtb * is_day_2 * is_ios,  'auc'),
            ('ios_half_rtb_rtr_d3',     p_rtr,    retention_label,     is_half_rtb * is_day_3 * is_ios,  'auc'),
            ('ios_half_rtb_rtr_d4',     p_rtr,    retention_label,     is_half_rtb * is_day_4 * is_ios,  'auc'),
            ('ios_half_rtb_rtr_d5',     p_rtr,    retention_label,     is_half_rtb * is_day_5 * is_ios,  'auc'),
            ('ios_half_rtb_rtr_d6',     p_rtr,    retention_label,     is_half_rtb * is_day_6 * is_ios,  'auc'),
            ('ios_half_rtb_rtr_d7',     p_rtr,    retention_label,     is_half_rtb * is_day_7 * is_ios,  'auc'),
        ]

    lt_log_loss = tf.losses.log_loss((valid_lt7 - 1) / 6.0, (lt7_pred - 1) / 6.0, tf.cast(tf.ones_like(is_rta), tf.float32), reduction="weighted_sum")
    
    my_step = config.get_step()
    TOPIC_ID = config.get_extra_param('TOPIC_ID_HOLIDAY')
    TOPIC_ID_Mean = tf.reduce_mean(TOPIC_ID)
    total_dense_var = config.get_dense_trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in total_dense_var])
    
    mask1 = tf.cast(valid_retention_label > 0.5, tf.float32) # 如果是1，则mask1=1，否则0
    mask0 = 1.0 - mask1 # 如果是0，则mask0=1，否则0
    m = 0.05
    pairwise_loss = (mask1 * tf.maximum((1 - m) * fusion_prtr - p_rtr, 0) + mask0 * tf.maximum(p_rtr - (1 + m) * fusion_prtr, 0)) * is_rtb
    pairwise_loss = tf.reduce_sum(pairwise_loss)

    tf.summary.histogram("rtr_loss", rtr_loss)
    tf.summary.histogram("lt2_loss", lt7_loss)
    tf.summary.histogram("l2_loss", l2_loss)
    tf.summary.histogram("lt_log_loss", lt_log_loss)

    train_step = config.get_step()
    #------------------------------开始打印----------------------------------

    # need_print_tensor_dict["step"] = train_step
    # need_print_tensor_dict["hidden_states"] = tf.shape(hmm1)
    # need_print_tensor_dict["mixing_h_hmm"] = tf.shape(mixing_h_hmm)
    # need_print_tensor_dict["feature_user_concatenated"] = tf.shape(feature_user_concatenated)
    # need_print_tensor_dict["feature_1_concatenated"] = tf.shape(feature_1_concatenated)
    # need_print_tensor_dict["feature_2_concatenated"] = tf.shape(feature_2_concatenated)
    # need_print_tensor_dict["final_concatenated_features"] = tf.shape(final_concatenated_features)
    need_print_tensor_dict[""] = tf.shape(final_concatenated_features)

    need_print_tensor_dict["batch_size"] = batch_size
    
    config.add_run_hook(TensorsPrintHook(need_print_tensor_dict), "tensor_print_hook")

    print_label_op = tf.cond(
        tf.equal(tf.mod(my_step, 100), 1),
        lambda: tf.print("step:", my_step, 'TOPIC_ID_Mean:',TOPIC_ID_Mean, "rtr_loss:", rtr_loss, "lt7_loss:", lt7_loss, 'l2_loss', l2_loss, "lt_log_loss:", lt_log_loss, "duration_loss:", duration_loss, "pairwise_loss:", pairwise_loss,
                         summarize=-1, output_stream=sys.stdout),
        lambda: tf.no_op())
    
    with tf.control_dependencies([print_label_op]):
        loss = rtr_loss + lt7_loss + lt_log_loss * 0.3 + l2_loss * 0.0 + duration_loss * 0.3 + pairwise_loss * 0.0
    
    # build model
    # config.set_feature_score_attr("is_retention", data_source_name="train")
    optimizers = []
    sparse_optimizer = config.optimizer.Adam(0.001)
    dense_optimizer = config.optimizer.Adam(0.001)
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
            'lt7': valid_lt7,
            'rta_rtr': p_rtr,
            'rtb_rtr': p_rtr,
            'half_rtb_rtr': p_rtr,
            'rta_lt7_pred': lt7_pred,
            'rtb_lt7_pred': lt7_pred,
            'half_rtb_lt7_pred': lt7_pred,
            'time_stamp': config.get_extra_param('time_stamp')
        }
        config.add_run_hook(DumpTensorHook('lt7_wangzihan', log_dict), 'custom_dump_tensor_hook')
else:
    targets = [
        ('rta2_rtr_is_holiday_type_0',     pred_probs_dict[0]),
        ('rta2_rtr_is_holiday_type_1',     pred_probs_dict[1]),
        ('rta2_rtr_is_holiday_type_2',     pred_probs_dict[2]),
        ('rta2_rtr_is_holiday_type_3',     pred_probs_dict[3]),
        # ('rta2_rtr_is_holiday_type_4',     pred_probs_dict[4]),
        # ('rta_rtr',      p_rtr),
        # ('rtb_rtr_is_holiday_type_0',     pred_probs_dict[0]),
        # ('rtb_rtr_is_holiday_type_1',     pred_probs_dict[1]),
        # ('rtb_rtr_is_holiday_type_2',     pred_probs_dict[2]),
        # ('rtb_rtr_is_holiday_type_3',     pred_probs_dict[3]),
        # ('rtb_rtr_is_holiday_type_4',     pred_probs_dict[4]),
        # ('rtb_rtr',      p_rtr),
        ('rta2_lt7',     lt7_pred),
        # ('rta_lt7',      lt7_pred),
        # ('rtb_lt7',      lt7_pred)
    ]
    q_names, preds = zip(*targets)
    config.dump_predict_config(
        'infer_conf/models/reflux_model_fusion_v1/',
        targets, input_type=3,
        extra_preds=q_names
    )
