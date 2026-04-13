---
name: ctr-training
description: 使用 FuxiCTR 框架训练各种 CTR 预测模型（DeepFM、DCN、DIN、DCNv2、xDeepFM 等）
---

# CTR 模型训练 Skill

本 skill 提供在 FuxiCTR 框架下配置、训练和评估各种 CTR 预测模型的完整流程。

## 支持的模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **DeepFM** | FM + DNN 结合 | 一般 CTR 预测 |
| **DCN/DCNv2** | 显式高阶特征交叉 | 需要特征交互建模 |
| **DIN** | 注意力机制 | 用户行为序列建模 |
| **xDeepFM** | CIN 压缩交互网络 | 高阶特征组合 |
| **AutoInt** | 自注意力机制 | 多特征交互 |
| **NFM** | Bi-Interaction Pooling | 二阶特征交互 |
| **AFM** | 注意力 FM | 重要特征加权 |

## 前置条件

- 已安装 FuxiCTR (`pip install fuxictr>=2.3.7`)
- 已准备数据集（参考 dataset-download skill）

## 快速开始

```bash
# 进入模型目录（以 DCNv2 为例，其他模型类似）
cd FuxiCTR/model_zoo/DCNv2

# 运行训练
python run_expid.py \
    --config ./config \
    --expid MyExperiment \
    --gpu -1
```

## 模型选择指南

### 根据数据特征选择

```
有用户行为序列？ → DIN / DIEN / BST
需要显式高阶交叉？ → DCN / DCNv2 / xDeepFM
一般 CTR 任务？ → DeepFM / Wide&Deep
特征数量少？ → FM / AFM / NFM
需要可解释性？ → AFM / xDeepFM
```

### 模型对比

| 模型 | 参数量 | 训练速度 | 精度 | 实现复杂度 |
|------|--------|----------|------|------------|
| FM | 低 | 快 | 中 | 简单 |
| DeepFM | 中 | 中 | 高 | 简单 |
| DCN | 中 | 中 | 高 | 中等 |
| DIN | 高 | 慢 | 高 | 复杂 |
| xDeepFM | 高 | 慢 | 高 | 复杂 |

## 配置文件说明

### 1. 数据集配置 (dataset_config.yaml)

路径：`FuxiCTR/model_zoo/{MODEL}/config/dataset_config.yaml`

```yaml
# 通用模板 - 根据实际数据集修改
dataset_name:
    data_format: csv
    data_root: /绝对路径/到/data/          # 数据根目录
    feature_cols:                           # 特征列定义
    - {active: true, dtype: str, name: user_id, type: categorical}
    - {active: true, dtype: str, name: item_id, type: categorical}
    - {active: true, dtype: str, name: category, type: categorical}
    label_col: {dtype: float, name: label}  # 标签列
    min_categr_count: 1                     # 最小类别计数阈值
    test_data: /绝对路径/到/data/test.csv
    train_data: /绝对路径/到/data/train.csv
    valid_data: /绝对路径/到/data/valid.csv
```

**特征类型：**

```yaml
# 分类特征
type: categorical
dtype: str

# 数值特征  
type: numeric
dtype: float
normalizer: MinMax  # 或 Standard

# 序列特征（用于 DIN/DIEN/BST）
type: sequence
dtype: str
max_len: 100
padding: pre
splitter: ^
share_embedding: item_id
```

### 2. 模型配置 (model_config.yaml)

路径：`FuxiCTR/model_zoo/{MODEL}/config/model_config.yaml`

```yaml
MyExperiment:                           # 实验ID（唯一）
    model: DCNv2                        # 模型名称
    dataset_id: dataset_name            # 对应 dataset_config.yaml
    loss: 'binary_crossentropy'
    metrics: ['AUC', 'logloss']
    task: binary_classification
    
    # 优化器
    optimizer: adam
    learning_rate: 1.0e-3
    
    # 嵌入层
    embedding_dim: 64
    embedding_regularizer: 0
    
    # 训练设置
    batch_size: 4096
    epochs: 10
    shuffle: True
    seed: 2024
    early_stop_patience: 2
    
    # 监控指标
    monitor: 'AUC'
    monitor_mode: 'max'
```

## 各模型配置示例

### DeepFM

```yaml
DeepFM_example:
    model: DeepFM
    dataset_id: my_dataset
    
    # DeepFM 特有
    hidden_units: [256, 128]
    net_dropout: 0.2
    batch_norm: True
    
    # 通用参数
    embedding_dim: 64
    batch_size: 4096
    learning_rate: 1.0e-3
    epochs: 10
```

### DCN / DCNv2

```yaml
DCNv2_example:
    model: DCNv2
    dataset_id: my_dataset
    
    # DCN 特有
    num_cross_layers: 3
    model_structure: parallel        # parallel 或 stacked
    use_low_rank_mixture: False
    low_rank: 32
    num_experts: 4
    
    # DNN 结构
    stacked_dnn_hidden_units: [256, 128]
    parallel_dnn_hidden_units: [256, 128]
    dnn_activations: relu
    net_dropout: 0.2
    batch_norm: True
    
    # 通用参数
    embedding_dim: 64
    batch_size: 4096
    learning_rate: 1.0e-3
    epochs: 10
```

### DIN（带注意力机制的序列模型）

```yaml
DIN_example:
    model: DIN
    dataset_id: my_dataset
    
    # DIN 特有
    attention_units: [64, 32]
    attention_activation: dice
    use_softmax: True
    
    # DNN 结构
    hidden_units: [256, 128, 64]
    net_dropout: 0.2
    batch_norm: True
    
    # 通用参数
    embedding_dim: 64
    batch_size: 4096
    learning_rate: 1.0e-3
    epochs: 10
```

### xDeepFM

```yaml
xDeepFM_example:
    model: xDeepFM
    dataset_id: my_dataset
    
    # xDeepFM 特有
    cin_layer_size: [128, 128]
    cin_direct: False
    cin_activation: relu
    hidden_units: [256, 128]
    net_dropout: 0.2
    batch_norm: True
    
    # 通用参数
    embedding_dim: 64
    batch_size: 4096
    learning_rate: 1.0e-3
    epochs: 10
```

### AutoInt

```yaml
AutoInt_example:
    model: AutoInt
    dataset_id: my_dataset
    
    # AutoInt 特有
    attention_layers: 3
    num_heads: 2
    attention_dim: 64
    hidden_units: [256, 128]
    net_dropout: 0.2
    attn_dropout: 0.1
    layer_norm: True
    use_residual: True
    
    # 通用参数
    embedding_dim: 64
    batch_size: 4096
    learning_rate: 1.0e-3
    epochs: 10
```

## 运行训练

### 基本命令

```bash
cd FuxiCTR/model_zoo/{MODEL}

# CPU 训练
python run_expid.py --config ./config --expid MyExperiment --gpu -1

# GPU 训练
python run_expid.py --config ./config --expid MyExperiment --gpu 0

# 多 GPU 训练
python run_expid.py --config ./config --expid MyExperiment --gpu 0,1,2,3
```

### 模型目录结构

```
FuxiCTR/model_zoo/
├── DeepFM/
│   ├── config/
│   │   ├── dataset_config.yaml
│   │   └── model_config.yaml
│   └── run_expid.py
├── DCNv2/
│   ├── config/
│   └── run_expid.py
├── DIN/
│   ├── config/
│   └── run_expid.py
├── xDeepFM/
│   ├── config/
│   └── run_expid.py
└── AutoInt/
    ├── config/
    └── run_expid.py
```

## 查看结果

### 实时查看日志

```bash
tail -f FuxiCTR/model_zoo/{MODEL}/checkpoints/{dataset}/{experiment}.log
```

### 日志输出示例

```
2026-04-10 13:37:00 INFO [Metrics] AUC: 0.958336 - logloss: 0.1423
2026-04-10 13:37:00 INFO Save best model: monitor(max)=0.958336
```

### 输出文件

```
checkpoints/
└── {dataset}/
    ├── {experiment}.log          # 训练日志
    └── {experiment}.model        # 最佳模型权重
```

## 超参数调优指南

### 学习率

```yaml
# 大数据集（千万级）
learning_rate: 1.0e-3

# 中等数据集（百万级）
learning_rate: 5.0e-4

# 小数据集（十万级）
learning_rate: 1.0e-4
```

### 批大小

```yaml
# 根据 GPU 显存调整
# 8GB 显存: 1024-2048
# 16GB 显存: 2048-4096
# 32GB 显存: 4096-8192
batch_size: 4096
```

### 嵌入维度

```yaml
# 特征规模大（>10万）
embedding_dim: 32

# 特征规模中等
embedding_dim: 64

# 特征规模小（<1万）且数据充足
embedding_dim: 128
```

### 网络深度

```yaml
# DCN 交叉层数
num_cross_layers: 2   # 小数据集
num_cross_layers: 3   # 中等数据集
num_cross_layers: 4   # 大数据集

# DNN 隐藏层
hidden_units: [256, 128]      # 通用
hidden_units: [512, 256, 128] # 复杂任务
hidden_units: [128, 64]       # 简单任务/小数据
```

## 故障排查

### 错误1：NumPy 2.0 兼容性

**症状：**
```
AttributeError: `np.Inf` was removed in the NumPy 2.0 release
```

**修复：**
```bash
sed -i 's/np.Inf/np.inf/g' \
  venv/lib/python*/site-packages/fuxictr/pytorch/models/rank_model.py
```

### 错误2：CUDA OOM

**症状：**
```
RuntimeError: CUDA out of memory
```

**修复：**
- 减小 `batch_size`
- 减小 `embedding_dim`
- 减小 `hidden_units`
- 使用 CPU 训练

### 错误3：模型配置不存在

**症状：**
```
RuntimeError: config_dir=xxx is not valid!
```

**修复：**
确保 `model_config.yaml` 中定义了对应的实验ID。

### 错误4：特征不匹配

**症状：**
```
KeyError: 'column_name'
```

**修复：**
检查 CSV 列名与 `feature_cols` 中的 `name` 是否完全一致。

## 高级用法

### 使用预训练嵌入

```yaml
feature_cols:
- name: item_id
  type: categorical
  pretrained_emb: /path/to/embedding.h5
  embedding_dim: 64
```

### 多任务学习

```yaml
# 使用多任务模型（MMoE、PLE 等）
model: MMoE
num_tasks: 2
num_experts: 8
task_layers: [64, 32]
```

### 自定义损失函数

```yaml
loss: 'focal_loss'  # 或自定义损失
focal_loss_gamma: 2.0
focal_loss_alpha: 0.25
```

## 性能对比参考

### MovieLensLatest 数据集

| 模型 | AUC | 训练时间 | 参数量 |
|------|-----|----------|--------|
| LR | 0.75 | 1 min | 50K |
| FM | 0.82 | 2 min | 100K |
| DeepFM | 0.95 | 5 min | 2M |
| DCNv2 | 0.96 | 6 min | 6M |
| DIN | 0.97 | 10 min | 8M |

### Frappe 数据集

| 模型 | AUC | 训练时间 | 参数量 |
|------|-----|----------|--------|
| FM | 0.90 | 1 min | 30K |
| DeepFM | 0.97 | 3 min | 1M |
| xDeepFM | 0.98 | 8 min | 3M |

## 相关资源

- [FuxiCTR GitHub](https://github.com/reczoo/FuxiCTR)
- [BARS Benchmark](https://github.com/reczoo/BARS)
- [模型论文汇总](https://github.com/reczoo/BARS/tree/main/papers)
