# =========================================================================
# DCN (Deep & Cross Network) 完整训练和评估示例
# 基于 FuxiCTR 框架
# =========================================================================

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import torch
from datetime import datetime

from fuxictr import datasets
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from model_zoo.DCN.DCN_torch.src import DCN


def train_and_evaluate_dcn():
    """
    完整的DCN模型训练和评估流程
    """
    print("=" * 60)
    print("DCN (Deep & Cross Network) 模型训练和评估")
    print("=" * 60)
    
    # 1. 配置参数
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, 'model_zoo/DCN/DCN_torch/config/')
    experiment_id = 'DCN_test'  # 对应配置文件中的实验ID
    
    # 切换到配置目录，以便正确加载相对路径的数据
    os.chdir(os.path.join(script_dir, 'model_zoo/DCN/DCN_torch/'))
    
    # 加载配置
    params = load_config(config_dir, experiment_id)
    params['gpu'] = -1  # 使用CPU训练，如有GPU可改为 0
    
    # 设置日志和随机种子
    set_logger(params)
    logging.info("=" * 60)
    logging.info("配置参数: " + print_to_json(params))
    seed_everything(seed=params['seed'])
    
    # 2. 加载特征映射
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("特征规格: " + print_to_json(feature_map.features))
    
    # 3. 初始化DCN模型
    print("\n" + "=" * 60)
    print("模型架构信息")
    print("=" * 60)
    model = DCN(feature_map, **params)
    model.count_parameters()  # 打印模型参数量
    
    # 4. 准备数据加载器
    print("\n" + "=" * 60)
    print("数据加载")
    print("=" * 60)
    train_gen, valid_gen = RankDataLoader(
        feature_map, 
        stage='train', 
        **params
    ).make_iterator()
    
    # 5. 模型训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    model.fit(train_gen, validation_data=valid_gen, **params)
    
    # 6. 验证集评估
    print("\n" + "=" * 60)
    print("验证集评估")
    print("=" * 60)
    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    
    # 清理内存
    del train_gen, valid_gen
    import gc
    gc.collect()
    
    # 7. 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(
            feature_map, 
            stage='test', 
            **params
        ).make_iterator()
        test_result = model.evaluate(test_gen)
    
    # 8. 打印最终结果
    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)
    print(f"验证集结果: {valid_result}")
    print(f"测试集结果: {test_result}")
    
    return valid_result, test_result


def demonstrate_dcn_model_structure():
    """
    展示DCN模型的结构
    """
    print("\n" + "=" * 60)
    print("DCN 模型结构说明")
    print("=" * 60)
    print("""
DCN (Deep & Cross Network) 模型结构:

1. 输入层 (Input Layer):
   - 输入特征经过 Embedding 层转换为稠密向量
   - embedding_dim: 嵌入维度

2. Cross Network (交叉网络):
   - 通过交叉层学习特征的高阶交互
   - num_cross_layers: 交叉层数
   - 每一层计算: x_{l+1} = x_0 * x_l^T * w_l + b_l + x_l

3. Deep Network (深度网络):
   - 标准的多层感知机 (MLP)
   - dnn_hidden_units: DNN隐藏层单元数, 如 [64, 32]
   - dnn_activations: 激活函数, 如 ReLU

4. 输出层 (Output Layer):
   - 将 Cross Network 和 Deep Network 的输出拼接
   - 经过线性层和 Sigmoid 激活函数输出预测概率

模型特点:
- 结合了显式的高阶特征交叉 (Cross Network) 和隐式的特征学习 (Deep Network)
- Cross Network 参数效率高, 每一层只有一个向量
- 适合处理稀疏的类别型特征
    """)


if __name__ == '__main__':
    # 展示模型结构
    demonstrate_dcn_model_structure()
    
    # 执行训练和评估
    valid_result, test_result = train_and_evaluate_dcn()
    
    print("\n" + "=" * 60)
    print("DCN 模型训练和评估完成!")
    print("=" * 60)
