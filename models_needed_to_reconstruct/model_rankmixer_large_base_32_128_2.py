from __future__ import print_function

import math
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
all_labels = ['retention_label', 'lt7', 'app_usage_duration_0d_bin','app_usage_duration_1d_bin','app_usage_duration_2d_bin', 'valid_duration_lt7', 'is_not_0vv_lt7', 'valid_retention_label',
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

def simple_dense_network_with_bn(inputs, units, name, act=tf.nn.relu, dropout = 0.0, batch_norm = False, training=False):
    output = inputs
    for i, unit in enumerate(units):
        output = mio_dense_layer_with_bn(output, unit, act, batch_norm, name='dense_{}_{}'.format(name, i), training=training)
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

def transformer_component(query_input, action_list_input, nh=4, att_emb_size=16, scope='attention'):
    with tf.variable_scope(scope):
        col = query_input.get_shape().as_list()[-1]
        action_item_size = action_list_input.get_shape().as_list()[-1]
        rown = tf.shape(query_input)[0]
        Q = tf.get_variable('q_trans_matrix', shape=(col, att_emb_size * nh))
        K = tf.get_variable('k_trans_matrix', shape=(action_item_size, att_emb_size * nh))
        V = tf.get_variable('v_trans_matrix', shape=(action_item_size, att_emb_size * nh))
        
        Q, K, V = map(lambda x: tf.tile(tf.expand_dims(x, 0), [rown, 1, 1]), (Q, K, V))
        querys = tf.matmul(query_input, Q)  															# [bs, seq_q, emb_size * nh]
        keys = tf.matmul(action_list_input, K)														# [bs, seq_len, emb_size * nh]
        values = tf.matmul(action_list_input, V)  												# [bs, seq_len, emb_size * nh]

        ## 以下部分的 pattern 会被优化脚本自动识别并替换
        querys = tf.stack(tf.split(querys, nh, axis=2))  									# [nh, bs, seq_q, emb_size]
        keys = tf.stack(tf.split(keys, nh, axis=2))  											# [nh, bs, seq_len, emb_size]
        values = tf.stack(tf.split(values, nh, axis=2))  									# [nh, bs, seq_len, emb_size]

        inner_product = tf.matmul(querys, keys, transpose_b=True) / 4.0  	# [nh, bs, seq_q, seq_len]
        normalized_att_scores = tf.nn.softmax(inner_product)  						# [nh, bs, seq_q, seq_len]
        result = tf.matmul(normalized_att_scores, values)  								# [nh, bs, seq_q, emb_size]
        result = tf.transpose(result, perm=[1, 2, 0, 3])  								# [bs, seq_q, nh, emb_size]
        result = tf.reshape(result, (-1, col * nh * att_emb_size))							# [bs, nh * emb_size]
    return result
    
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
            # out = simple_dense_network_with_ppnet(input_embedding, ppnet_input, dnn_layers, ppnet_unit, name="cond_prob_{}".format(_name), act=tf.nn.leaky_relu, dropout=0.0, batch_norm=False, training=training)
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

def retain_everyday_with_ppnet(input_embedding,
                     ppnet_input,
                     dnn_layers=[16],
                     ppnet_unit=16,
                     probs=7,
                     scope='default',
                     residual_logit=None,
                     training=False):
    pred_probs_dict = {}
    with tf.variable_scope("retain_everyday_{}".format(scope)):
        for i in range(probs):
            _name = "{}".format(i)
            out = simple_dense_network_with_ppnet(input_embedding, ppnet_input, dnn_layers, ppnet_unit, name="prob_{}".format(_name), act=tf.nn.relu, dropout=0.0, batch_norm=False, training=training)
            # out = simple_dense_network_with_ppnet(input_embedding, ppnet_input, dnn_layers, ppnet_unit, name="prob_{}".format(_name), act=tf.nn.leaky_relu, dropout=0.0, batch_norm=False, training=training)
            out = mio_dense_layer(out, 1, activation=None, name="prob_{}_last_fc".format(_name))
            if residual_logit is not None:
                pred_probs_dict[i] = tf.nn.sigmoid(out+residual_logit[i])
            else:
                pred_probs_dict[i] = tf.nn.sigmoid(out)

        pred_probs_concat = tf.reshape(tf.concat([pred_probs_dict[i] for i in range(probs)], axis=1), (-1, probs))

    return {
        "pred_probs_dict": pred_probs_dict,
        "pred_probs_concat": pred_probs_concat
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


def mmoe_layer(inputs, expert_units, name, num_experts, num_tasks, expert_act=tf.nn.relu, gate_act=tf.nn.softmax):
    expert_outputs, final_outputs = [], []
    with tf.name_scope('experts_network'):
        for i in range(num_experts):
            weight_name_template = name + '_expert{}_'.format(i) + 'param'
            expert_layer = simple_dense_network(inputs, expert_units, weight_name_template, act=expert_act)
            expert_outputs.append(expand_dims(expert_layer, axis=2))
        expert_outputs = tf.concat(expert_outputs, 2)              #(batch_size, expert_units[-1], num_experts)

    with tf.name_scope('gates_network'):
        for i in range(num_tasks):
            weight_name_template = name + '_task_gate{}_'.format(i) + 'param'
            gate_layer = mio_dense_layer(inputs, num_experts, gate_act, weight_name_template)
            expanded_gate_output = expand_dims(gate_layer, axis=1)  #(batch_size, ?, num_experts)
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, expert_units[-1], axis=1)
            final_outputs.append(backend_sum(weighted_expert_output, axis=2))
    return final_outputs #(num_tasks, batch_size, expert_units[-1])


def calc_contrastive_loss(pred_prob, pred_lt7, valid_index):
    # 计算对比损失
    # pred_prob: 预估的概率p(lt>=k)
    # pred_lt7: 预估的lt7
    # valid_index: 参与计算对比损失的有效样本，(投放方式)*(lt>=k)

    valid_num = tf.cast(tf.reduce_sum(valid_index), tf.int32)
    rows, cols = tf.unstack(tf.where(valid_index > 0), axis=1, num=2)

    pred_prob_valid = tf.gather(pred_prob, rows)
    pred_lt7_valid = tf.gather(pred_lt7, rows)
    pred_lt7_trans = tf.math.log(pred_lt7_valid + 1.0) / tf.math.log(10.0) # lt7映射

    pred_prob_matrix = tf.reshape(pred_prob_valid, [valid_num, -1]) - tf.reshape(pred_prob_valid, [-1, valid_num])
    pred_lt7_matrix = tf.reshape(pred_lt7_trans, [valid_num, -1]) - tf.reshape(pred_lt7_trans, [-1, valid_num])

    loss_contrastive = -tf.reduce_sum(pred_prob_matrix * pred_lt7_matrix) / (tf.cast(valid_num, tf.float32) * tf.cast(valid_num, tf.float32))

    return loss_contrastive


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

def linear_anneal_until(step, start=1.0, end=0.05, anneal_steps=10000, dtype=tf.float32):
    step_f = tf.cast(step, dtype)
    S = tf.cast(anneal_steps, dtype)
    p = tf.minimum(step_f / S, 1.0)  # step>=S 时 p=1
    return tf.cast(start, dtype) + (tf.cast(end, dtype) - tf.cast(start, dtype)) * p

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
    
def manual_logsumexp(logits, axis=2, keepdims=True):
    """
    公式: log(sum(exp(x))) = m + log(sum(exp(x - m))), 其中 m = max(x)
    """
    logits_max = tf.reduce_max(logits, axis=axis, keepdims=True)
    centered_logits = logits - logits_max
    exp_logits = tf.exp(centered_logits)
    sum_exp = tf.reduce_sum(exp_logits, axis=axis, keepdims=keepdims)
    log_sum = tf.log(sum_exp)
    result = log_sum + logits_max
    return result

def log_sinkhorn(logits, n_iters=20, temperature=0.1, step=None,
                 temp_start=1.0, temp_end=0.05, anneal_steps=10000):
    original_shape = tf.shape(logits)

    logits = tf.cond(
        pred=tf.equal(tf.rank(logits), 2),
        true_fn=lambda: tf.expand_dims(logits, axis=0),  # [K, K] -> [1, K, K]
        false_fn=lambda: logits                          # 保持 [N, K, K]
    )

    if step is not None:
        temperature = linear_anneal_until(step, start=temp_start, end=temp_end,
                                          anneal_steps=anneal_steps, dtype=logits.dtype)

    temperature = tf.maximum(tf.cast(temperature, logits.dtype), tf.cast(1e-6, logits.dtype))
    logits = logits / temperature

    for _ in range(n_iters):
        # logits = logits - tf.reduce_logsumexp(logits, axis=2, keepdims=True)
        # logits = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
        logits = logits - manual_logsumexp(logits, axis=2, keepdims=True)
        logits = logits - manual_logsumexp(logits, axis=1, keepdims=True)
    
    res = tf.exp(logits)
    return tf.reshape(res, original_shape)

def tiled_identity_initializer(shape, dtype=tf.float32, partition_info=None):
    n_chunks = shape[0]
    k = shape[1]
    I = tf.eye(k, dtype=dtype)
    I_expanded = tf.reshape(I, [1, k, k])
    
    return tf.tile(I_expanded, [n_chunks, 1, 1])

def commutation_matrix_initializer(T):
    """
    创建一个自定义初始化器，返回 T^2 x T^2 的交换矩阵 K_{T,T}。
    该矩阵对应于将 shape 为 [T, T] 的矩阵进行转置的操作。
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        I_T2 = tf.eye(T * T, dtype=dtype)
        temp = tf.reshape(I_T2, [T, T, T * T])
        temp = tf.transpose(temp, perm=[1, 0, 2])

        K_TT = tf.reshape(temp, [T * T, T * T])
        return K_TT
    return _initializer

def efficient_kron_interaction_doubly_stochastic(x, block_size, sinkhorn_iters=20):
    emb_dim = x.get_shape().as_list()[1]
    if emb_dim % block_size != 0:
        raise ValueError(f"emb_dim ({emb_dim}) must be divisible by block_size ({block_size})")
    
    N = emb_dim // block_size          # Chunks
    K = block_size         # Chunk dim
    
    A_logits = tf.get_variable(
        "Matrix_A_logits", 
        shape=[N, N],
        initializer=commutation_matrix_initializer(16)
    )
    
    W_logits = tf.get_variable(
        "Matrices_Wi", 
        shape=[N, K, K],
        initializer=tiled_identity_initializer
    )

    #————————————————————全局/局部参数矩阵的对称约束————————————————————
    A_logits_constraint = (A_logits + tf.transpose(A_logits, perm=[1, 0])) * 0.5
    W_logits_constraint = (W_logits + tf.transpose(W_logits, perm=[0, 2, 1])) * 0.5

    if training:
        A = log_sinkhorn(A_logits_constraint, n_iters=sinkhorn_iters, step=step,
                         temp_start=1.0, temp_end=0.05, anneal_steps=10000)
        W = log_sinkhorn(W_logits_constraint, n_iters=sinkhorn_iters, step=step,
                         temp_start=1.0, temp_end=0.05, anneal_steps=10000)
    else:
        # predict 固定温度
        A = log_sinkhorn(A_logits_constraint, n_iters=sinkhorn_iters, temperature=0.05)
        W = log_sinkhorn(W_logits_constraint, n_iters=sinkhorn_iters, temperature=0.05)
    x_reshaped = tf.reshape(x, [-1, N, K])
    x_local = tf.einsum('bni, nio -> bno', x_reshaped, W)
    x_global = tf.einsum('mn, bno -> bmo', A, x_local)
    
    output = tf.reshape(x_global, [-1, emb_dim])
    return output

def RankMixer_simplified(hidden_states, d_model, block_size, d_ff, scope="RankMixer", is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        l = hidden_states.get_shape().as_list()[1]

        # # TokenMixing
        # mixing_h = tf.reshape(hidden_states, [-1, l, l, d_model // l]) # d_model = 512
        # mixing_h = tf.transpose(mixing_h, perm=[0, 2, 1, 3])
        # mixing_h = tf.reshape(mixing_h, [-1, l, d_model]) # [batch_size, T, D]
        # mixing_h = tf.reshape(mixing_h, [-1, l * d_model])

        # 转置
        mixing_h = tf.transpose(hidden_states, perm=[0, 2, 1])
        mixing_h = tf.reshape(mixing_h, [-1, l * d_model])

        # mixing_h = tf.reshape(hidden_states, [-1, l * d_model])
        mixing_h = efficient_kron_interaction_doubly_stochastic(mixing_h, block_size)
        mixing_h = tf.reshape(mixing_h, [-1, l, d_model])
        
        mixed_hidden_states = rms_norm(hidden_states + mixing_h, scope="post_mixing_norm")
        
        all_token_outputs = []
        for i in range(l):
            token_slice = tf.slice(mixed_hidden_states, [0, i, 0], [-1, 1, -1])  # [B, 1, D]
            
            with tf.variable_scope(f"Token_{i}_FFN", reuse=reuse):
                token_output = ffn(token_slice, d_model, d_ff)
            
            all_token_outputs.append(token_output)

        ffn_output = tf.concat(all_token_outputs, axis=1)
        
        final_hidden_states = rms_norm(mixed_hidden_states + ffn_output, scope="post_ffn_norm")

        return final_hidden_states

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

        gate = mio_dense_layer(reshaped_input, d_ff_glu, activation=None, name=f'{scope}_gate_proj')
        up = mio_dense_layer(reshaped_input, d_ff_glu, activation=None, name=f'{scope}_up_proj')

        activated_up = swish(gate) * up

        # output = mio_dense_layer(activated_up, d_model, activation=None, name=f'{scope}_down_proj')
        # 备选方案：直接使用 tf.layers.dense
        output = tf.layers.dense(
            activated_up,
            d_model,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(scale=0.01, mode='fan_avg', distribution='uniform'),
            name=f'{scope}_down_proj'
        )
        
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
            expert_output = ffn_swish_glu(token_input, d_model, d_ff, f"{scope}_ffn{i}_")
            weighted_expert_output = gate_splits[i] * expert_output
            output += weighted_expert_output

        return gate, output, load_balance_loss
    
def layer_norm(x, eps=1e-6, scope=None):
    """
    Layer Normalization - 对每个token的最后一个维度进行归一化
    
    Args:
        x: 输入tensor，shape为 [B, T, D] 或 [B, T, ..., D]
        eps: 防止除零的小常数
        scope: 变量作用域
    
    Returns:
        归一化后的tensor，shape与输入相同
    """
    with tf.variable_scope(scope or "layer_norm"):
        # 获取最后一个维度的大小
        layer_size = x.get_shape().as_list()[-1]
        
        # 可学习的缩放和平移参数
        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", [layer_size], initializer=tf.zeros_initializer())
        
        # 计算均值和方差（在最后一个维度上）
        mean = tf.reduce_mean(x, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keep_dims=True)
        
        # 归一化
        normalized = (x - mean) * tf.rsqrt(variance + eps)
        
        # 缩放和平移
        return normalized * scale + offset
    
def RankMixer_large(hidden_states, d_model, d_ff, expert_num, k, scope="RankMixer_large", denseFFN=False, is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        b, t, d = hidden_states.get_shape().as_list()
        l = t

        # =================================================================
        # Part 1: Mixing & Reverting (Pre-Norm)
        # =================================================================
        
        # 1. [Pre-Norm] 进入计算分支前先 Norm
        # normed_input_1 = layer_norm(hidden_states, scope="pre_attention_ln")

        normed_input_1 = rms_norm(hidden_states, scope="pre_mixing_norm")
        # 2. 计算分支：Mixing -> MoE -> Reverting
        # 注意：这里所有操作都基于 normed_input_1
        mixing_h = tf.reshape(normed_input_1, (-1, t, l, d // l))
        mixing_h = tf.transpose(mixing_h, perm=[0, 2, 1, 3])
        middle_dim = d // l * t
        mixing_h = tf.reshape(mixing_h, (-1, l, middle_dim))
        
        all_token_outputs = []
        all_gate_outputs = []
        total_load_balance_loss_1 = 0.0

        for i in range(l):
            token_slice = tf.slice(mixing_h, [0, i, 0], [-1, 1, -1])
            with tf.variable_scope(f"Mix_Token_{i}_MoE"): # 建议区分scope
                gate, token_output, load_balance_loss = SparseMoe_NoNorm(
                    1, token_slice, middle_dim, d_ff, expert_num, k, 
                    is_training=is_training, scope="l_moe"
                )
            
            all_token_outputs.append(token_output)
            all_gate_outputs.append(gate)
            total_load_balance_loss_1 += load_balance_loss

        moe_output_1 = tf.concat(all_token_outputs, axis=1)
        gates_1 = tf.concat(all_gate_outputs, axis=1)
        
        # Revert
        revert_h = tf.reshape(moe_output_1, (-1, l, t, middle_dim // t))
        revert_h = tf.transpose(revert_h, perm=[0, 2, 1, 3])
        reverted_output = tf.reshape(revert_h, (-1, t, d))
        
        # 3. [Residual] 关键修改！
        # 必须加在原始 hidden_states 上，且加完后不做 Norm
        middle_hidden_states = hidden_states + reverted_output

        # =================================================================
        # Part 2: Pertoken MoE (Pre-Norm)
        # =================================================================

        # 1. [Pre-Norm] 对上一阶段的输出做 Norm
        normed_input_2 = rms_norm(middle_hidden_states, scope="pre_moe_norm")

        all_token_outputs = []
        all_gate_outputs = []
        total_load_balance_loss_2 = 0.0

        # 2. 计算分支：Standard MoE
        # 注意：这里切片必须源自 normed_input_2
        for i in range(t):
            token_slice = tf.slice(normed_input_2, [0, i, 0], [-1, 1, -1])
            with tf.variable_scope(f"Std_Token_{i}_MoE"): # 建议区分scope
                gate, token_output, load_balance_loss = SparseMoe_NoNorm(
                    1, token_slice, d_model, d_ff, expert_num, k, 
                    is_training=is_training, scope="t_moe"
                )
            
            all_token_outputs.append(token_output)
            all_gate_outputs.append(gate)
            total_load_balance_loss_2 += load_balance_loss

        moe_output_2 = tf.concat(all_token_outputs, axis=1)
        gates_2 = tf.concat(all_gate_outputs, axis=1)
        
        # 3. [Residual] 关键修改！
        # 加在 Part 1 的输出(未Norm)上
        final_hidden_states = normed_input_2  + moe_output_2

        # 汇总 Loss (平均化)
        avg_load_balance_loss = (total_load_balance_loss_1 / l + total_load_balance_loss_2 / t) / 2.0

        return final_hidden_states

def wukong_linear_compress_block(inputs, num_emb_out, name="lcb"):
    with tf.variable_scope(name):
        num_emb_in = inputs.get_shape().as_list()[1]
        weight = tf.get_variable("weight", shape=(num_emb_in, num_emb_out), initializer=tf.glorot_uniform_initializer())
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, (0, 2, 1))
        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = tf.matmul(outputs, weight)
        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = tf.transpose(outputs, (0, 2, 1))
        return outputs

def wukong_factorization_machine_block(inputs, num_emb_out, rank, dim_hidden, dropout, is_training, name="fmb"):
    with tf.variable_scope(name):
        num_emb_in = inputs.get_shape().as_list()[1]
        dim_emb = inputs.get_shape().as_list()[2]
        
        weight = tf.get_variable("weight", shape=(num_emb_in, rank), initializer=tf.glorot_uniform_initializer())
        
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = tf.transpose(inputs, (0, 2, 1))
        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = tf.matmul(outputs, weight)
        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = tf.matmul(inputs, outputs)
        
        # 展平送入 MLP
        outputs = tf.reshape(outputs, (-1, num_emb_in * rank))
        outputs = rms_norm(outputs, scope="norm")
        
        # MLP: 隐藏层 + 线性输出层
        outputs = simple_dense_network(outputs, [dim_hidden], "mlp_hidden", act=tf.nn.relu, dropout=dropout if is_training else 0.0)
        outputs = mio_dense_layer(outputs, num_emb_out * dim_emb, activation=None, name="mlp_out")
        
        outputs = tf.reshape(outputs, (-1, num_emb_out, dim_emb))
        return outputs

def wukong_layer(inputs, config, is_training, name="wukong_layer"):
    with tf.variable_scope(name):
        num_emb_in = inputs.get_shape().as_list()[1]
        dim_emb = inputs.get_shape().as_list()[2]
        
        lcb_out = wukong_linear_compress_block(inputs, config.num_emb_lcb, name="lcb")
        fmb_out = wukong_factorization_machine_block(inputs, config.num_emb_fmb, config.rank_fmb, config.dim_hidden, config.dropout, is_training, name="fmb")
        
        # 拼接 (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = tf.concat([fmb_out, lcb_out], axis=1)
        
        # 残差对齐（如果输入输出特征序列长度不一致）
        if num_emb_in != (config.num_emb_lcb + config.num_emb_fmb):
            with tf.variable_scope("residual_projection"):
                res_weight = tf.get_variable("weight", shape=(num_emb_in, config.num_emb_lcb + config.num_emb_fmb), initializer=tf.glorot_uniform_initializer())
                res_out = tf.transpose(inputs, (0, 2, 1))
                res_out = tf.matmul(res_out, res_weight)
                res_out = tf.transpose(res_out, (0, 2, 1))
        else:
            res_out = inputs
            
        outputs = outputs + res_out
        outputs = rms_norm(outputs, scope="post_wukong_norm")
        return outputs
    
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
        self.token_num = 32 # origin = 32
        self.d_kv =  128 # origin = 512 每个head的维度，head数量等于token数量
        self.block_size = 64
        self.num_block = 2 #层数
        self.d_ff_coefficient = 2 # ffn中间层维度倍数，默认2倍
        self.expert_num = 2 #专家数量
        self.k = 1 #TopK
        self.alpha = 0.0 #负载均衡损失函数权重

class WukongConfig:
    def __init__(self):
        self.num_layers = 8         # 网络层数，可对齐原来的 mixer_config.num_block
        self.dim_emb = 1024         # 必须等于 token 投影后的维度 (d_kv)
        self.num_emb_lcb = 8        # LCB 输出的特征域数量
        self.num_emb_fmb = 8        # FMB 输出的特征域数量
        self.rank_fmb = 32          # FMB 低秩矩阵的秩
        self.dim_hidden = 1024      # FMB 内部 MLP 的隐藏层维度
        self.dropout = 0.1          # dropout 比例

wukong_config = WukongConfig()

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
# rate_mix_embedding = mio_dense_layer(rate_mix_embedding, 32, tf.nn.sigmoid, name='rate_mix_mlp')

mlp_ppnet_input_slots = [50, 51, 52, 59, 89]
mlp_ppnet_input_embs = config.new_embedding("mlp_ppnet_input_embs", dim=input_emb_dim, slots=mlp_ppnet_input_slots, **get_predict_param())

# mmoe_input = tf.concat([user_common_embs, request_feature_embs, tense_embs, active_app_emb, install_app_emb,
#                     channel_mix, usergroup_mix_embedding, colossus_feature_embedding, mlp_ppnet_input_embs], axis=1)
promotion_id_emb = config.new_embedding("promotion_id_emb", dim=input_emb_dim, slots=[7])
channel_type_emb = config.new_embedding("channel_type_emb", dim=input_emb_dim, slots=[8])
# creative_id_emb = config.new_embedding("creative_id_emb", dim=input_emb_dim, slots=[6])
rtb_feature_item = config.new_embedding("rtb_feature_item", dim=input_emb_dim, slots=rtb_feature_slots_item)

# 底层输入特征的门控输入特征
# emb_ppnet_input = tf.stop_gradient(tf.concat([tense_embs], axis=1))
# 顶层次留存模型的门控输入特征
mlp_ppnet_input = tf.stop_gradient(tf.concat([request_feature_embs, channel_mix, channel_type_emb, mlp_ppnet_input_embs], axis=1))

# input = tf.concat([mmoe_input, promotion_id_emb, channel_type_emb, rate_mix_embedding, rtb_feature, rtb_feature_item], axis=1)
# input = tf.multiply(simple_lhuc_network(emb_ppnet_input, 1024, input.get_shape().as_list()[-1], 'ppnet_emb'), input)

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
final_concatenated_features = tf.concat(feature_list, axis=1)

total_input_dim = final_concatenated_features.get_shape()[-1]
remainder = total_input_dim % mixer_config.token_num
if remainder != 0:
    padding_size = mixer_config.token_num - remainder
    batch_size = tf.shape(final_concatenated_features)[0]
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

hidden_states = tf.stack(all_processed_tokens, axis=1)

with tf.variable_scope("RankMixer_large_backbone"):
    mixing_ffn_dim = mixer_config.d_kv * mixer_config.d_ff_coefficient
    expert_num = mixer_config.expert_num
    mixer_regloss = 0

    hmm1 = hidden_states

    for i in range(mixer_config.num_block):
        hidden_states= RankMixer_large(
            hidden_states,
            mixer_config.d_kv,
            mixing_ffn_dim,
            mixer_config.expert_num,
            mixer_config.k,
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
    # pred_probs = shared_bottom_rtr_tower(input_embedding=rankmixer_output, ppnet_input=mlp_ppnet_input, 
    #                                      gating_input=ple_gating_input, tower_units=tower_units, 
    #                                      ppnet_unit=64, num_tasks=4, scope='rtr_tower', training=training)
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
    # #------------------------------开始打印----------------------------------

    # need_print_tensor_dict["step"] = train_step
    # need_print_tensor_dict["hmm1"] = tf.shape(hmm1)
    # need_print_tensor_dict["batch_size"] = batch_size
    
    # config.add_run_hook(TensorsPrintHook(need_print_tensor_dict), "tensor_print_hook")

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
