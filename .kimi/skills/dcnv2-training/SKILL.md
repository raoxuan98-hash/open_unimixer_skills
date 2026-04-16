---
name: dcnv2-training
description: 使用 FuxiCTR 框架训练和评估 DCNv2 CTR 预测模型
---

# DCNv2 模型训练 Skill

本 skill 提供在 FuxiCTR 框架下配置、训练和评估 DCNv2 模型的完整流程。

## 前置条件

- 已安装 FuxiCTR (`pip install fuxictr>=2.3.7`)
- 已准备数据集（参考 dataset-download skill）

## 快速开始

```bash
# 进入模型目录
cd FuxiCTR/model_zoo/DCNv2

# 运行训练
python run_expid.py \
    --config ./config \
    --expid DCNv2_test \
    --gpu -1
```

## 配置文件说明

### 1. 数据集配置 (dataset_config.yaml)

路径：`FuxiCTR/model_zoo/DCNv2/config/dataset_config.yaml`

```yaml
# 示例：MovielensLatest_x1 配置
movielenslatest_x1:
    data_format: csv
    data_root: /绝对路径/到/data/          # 数据根目录
    feature_cols:                           # 特征列定义
    - {active: true, dtype: str, name: user_id, type: categorical}
    - {active: true, dtype: str, name: item_id, type: categorical}
    - {active: true, dtype: str, name: tag_id, type: categorical}
    label_col: {dtype: float, name: label}  # 标签列
    min_categr_count: 1                     # 最小类别计数阈值
    test_data: /绝对路径/到/data/MovielensLatest_x1/test.csv
    train_data: /绝对路径/到/data/MovielensLatest_x1/train.csv
    valid_data: /绝对路径/到/data/MovielensLatest_x1/valid.csv
```

**关键参数：**

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `data_root` | 数据根目录（建议绝对路径） | `/home/user/project/data/` |
| `feature_cols` | 特征列列表 | 见下方详细说明 |
| `label_col` | 标签列定义 | `{dtype: float, name: label}` |
| `min_categr_count` | 过滤低频特征的阈值 | `1` 或 `10` |

**特征类型：**

```yaml
# 分类特征
type: categorical
dtype: str

# 数值特征  
type: numeric
dtype: float

# 序列特征（如用户历史行为）
type: sequence
dtype: str
max_len: 100                    # 最大序列长度
padding: pre                    # 填充位置
splitter: ^                     # 分隔符
share_embedding: item_id        # 共享嵌入
```

### 2. 模型配置 (model_config.yaml)

路径：`FuxiCTR/model_zoo/DCNv2/config/model_config.yaml`

```yaml
DCNv2_my_experiment:                    # 实验ID（唯一）
    model: DCNv2
    dataset_id: movielenslatest_x1      # 对应 dataset_config.yaml 中的名称
    loss: 'binary_crossentropy'
    metrics: ['AUC', 'logloss']
    task: binary_classification
    
    # 优化器设置
    optimizer: adam
    learning_rate: 1.0e-3
    
    # 模型结构
    model_structure: parallel           # parallel 或 stacked
    num_cross_layers: 3                 # 交叉网络层数
    use_low_rank_mixture: False
    low_rank: 32
    num_experts: 4
    
    # 嵌入层
    embedding_dim: 64
    embedding_regularizer: 0
    
    # DNN 结构
    stacked_dnn_hidden_units: [256, 128]
    parallel_dnn_hidden_units: [256, 128]
    dnn_activations: relu
    net_dropout: 0.2
    batch_norm: True
    net_regularizer: 0
    
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

**DCNv2 关键超参数：**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `num_cross_layers` | 交叉网络层数 | 2-4 |
| `embedding_dim` | 嵌入维度 | 32, 64, 128 |
| `model_structure` | 模型结构 | `parallel` 或 `stacked` |
| `batch_size` | 批大小 | 1024-8192 |
| `learning_rate` | 学习率 | 1e-3 ~ 1e-4 |

## 运行训练

### 基本命令

```bash
cd FuxiCTR/model_zoo/DCNv2

# CPU 训练
python run_expid.py --config ./config --expid DCNv2_my_experiment --gpu -1

# GPU 训练
python run_expid.py --config ./config --expid DCNv2_my_experiment --gpu 0

# 指定多个 GPU
python run_expid.py --config ./config --expid DCNv2_my_experiment --gpu 0,1,2,3
```

### 命令参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--config` | 是 | 配置文件目录路径 |
| `--expid` | 是 | 实验ID（对应 model_config.yaml 中的配置名） |
| `--gpu` | 否 | GPU 设备ID，-1表示CPU，默认-1 |

## 查看结果

### 实时查看日志

```bash
# 训练时实时查看
tail -f FuxiCTR/model_zoo/DCNv2/checkpoints/数据集名/实验ID.log

# 示例
tail -f checkpoints/movielenslatest_x1/DCNv2_my_experiment.log
```

### 日志输出示例

```
2026-04-10 13:37:00 INFO [Metrics] AUC: 0.958336 - logloss: 0.xxxxx
2026-04-10 13:37:00 INFO Save best model: monitor(max)=0.958336
```

### 输出文件

```
checkpoints/
└── 数据集名/
    ├── 实验ID.log          # 训练日志
    └── 实验ID.model        # 保存的最佳模型
```

## 不同数据集配置模板

### MovielensLatest_x1

```yaml
# dataset_config.yaml
movielenslatest_x1:
    data_format: csv
    data_root: /path/to/data/
    feature_cols:
    - {active: true, dtype: str, name: user_id, type: categorical}
    - {active: true, dtype: str, name: item_id, type: categorical}
    - {active: true, dtype: str, name: tag_id, type: categorical}
    label_col: {dtype: float, name: label}
    min_categr_count: 1
    test_data: /path/to/data/MovielensLatest_x1/test.csv
    train_data: /path/to/data/MovielensLatest_x1/train.csv
    valid_data: /path/to/data/MovielensLatest_x1/valid.csv

# model_config.yaml
DCNv2_movielens:
    model: DCNv2
    dataset_id: movielenslatest_x1
    batch_size: 4096
    embedding_dim: 64
    num_cross_layers: 3
    epochs: 10
    # ... 其他参数
```

### Frappe_x1

```yaml
# dataset_config.yaml
frappe_x1:
    data_format: csv
    data_root: /path/to/data/
    feature_cols:
    - {active: true, dtype: str, name: user, type: categorical}
    - {active: true, dtype: str, name: item, type: categorical}
    - {active: true, dtype: str, name: daytime, type: categorical}
    - {active: true, dtype: str, name: weekday, type: categorical}
    - {active: true, dtype: str, name: isweekend, type: categorical}
    - {active: true, dtype: str, name: homework, type: categorical}
    - {active: true, dtype: str, name: cost, type: categorical}
    - {active: true, dtype: str, name: weather, type: categorical}
    - {active: true, dtype: str, name: country, type: categorical}
    - {active: true, dtype: str, name: city, type: categorical}
    label_col: {dtype: float, name: label}
    # ... 数据路径
```

## 故障排查

### 错误1：NumPy 2.0 兼容性

**症状：**
```
AttributeError: `np.Inf` was removed in the NumPy 2.0 release
```

**修复：**
```bash
# 编辑 fuxictr 源码
sed -i 's/np.Inf/np.inf/g' \
  venv/lib/python*/site-packages/fuxictr/pytorch/models/rank_model.py
```

### 错误2：数据路径无效

**症状：**
```
AssertionError: Invalid data path: ../data/xxx/train.csv
```

**修复：**
在 `dataset_config.yaml` 中使用**绝对路径**。

### 错误3：OOM (内存不足)

**症状：**
```
RuntimeError: CUDA out of memory
```

**修复：**
- 减小 `batch_size`
- 减小 `embedding_dim`
- 使用 CPU 训练（`--gpu -1`）

### 错误4：特征列不匹配

**症状：**
```
KeyError: 'column_name'
```

**修复：**
检查 CSV 文件的列名与 `feature_cols` 中的 `name` 是否完全一致（区分大小写）。

## 高级用法

### 使用预训练嵌入

```yaml
feature_cols:
- name: item_emb
  type: categorical
  pretrained_emb: /path/to/item_visual_emb_dim64.h5
  embedding_dim: 64
```

### 序列特征配置

```yaml
feature_cols:
- name: pos_items
  type: sequence
  max_len: 100
  splitter: ^
  padding: pre
  share_embedding: item_id
```

### 多 GPU 训练

```bash
# 使用 DataParallel
python run_expid.py --config ./config --expid EXP --gpu 0,1,2,3
```

## 相关资源

- [FuxiCTR GitHub](https://github.com/reczoo/FuxiCTR)
- [DCNv2 论文](https://arxiv.org/abs/2008.13535)
- [BARS Benchmark](https://github.com/reczoo/BARS)
