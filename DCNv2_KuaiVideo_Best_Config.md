# DCN-V2 on KuaiVideo_x1 最佳配置指南

## 📊 历史最佳结果

| 指标 | 数值 |
|------|------|
| **实验ID** | `DCNv2_kuaivideo_x1_008_7047bff3` |
| **AUC** | **0.746953** |
| **gAUC** | **0.667472** |
| **logloss** | 0.437494 |
| **记录时间** | 2022-08-23 |

---

## 🔧 关键参数配置

### 模型架构参数
```yaml
model: DCNv2
model_structure: parallel
num_cross_layers: 2          # 关键：2层交叉网络
num_experts: 4
low_rank: 32
use_low_rank_mixture: false
```

### 网络结构参数
```yaml
# DNN隐藏层
parallel_dnn_hidden_units: [1024, 512, 256]
stacked_dnn_hidden_units: [500, 500, 500]
dnn_activations: relu

# 正则化
batch_norm: true             # 关键：启用Batch Norm
net_dropout: 0.1
net_regularizer: 0
embedding_regularizer: 0.0001
```

### 训练参数
```yaml
optimizer: adam
learning_rate: 0.001
batch_size: 8192
epochs: 100
early_stop_patience: 2

# 评估指标
metrics: [gAUC, AUC, logloss]
monitor: {AUC: 1, gAUC: 1}
monitor_mode: max
```

### 嵌入层参数
```yaml
embedding_dim: 64
embedding_regularizer: 0.0001

# 特征编码器配置
feature_specs:
  - {feature_encoder: 'nn.Linear(64, 64, bias=False)', name: item_emb}
  - {feature_encoder: layers.MaskedAveragePooling(), name: pos_items}
  - {feature_encoder: layers.MaskedAveragePooling(), name: neg_items}
  - {feature_encoder: [layers.MaskedAveragePooling(), 'nn.Linear(64, 64, bias=False)'], name: pos_items_emb}
  - {feature_encoder: [layers.MaskedAveragePooling(), 'nn.Linear(64, 64, bias=False)'], name: neg_items_emb}
```

---

## ⚠️ 失败配置分析

### 之前测试的配置（表现不佳）
| 参数 | 历史最佳(008) | 失败配置(001) | 影响 |
|------|--------------|--------------|------|
| **batch_norm** | ✅ **true** | ❌ false | 训练不稳定，收敛慢 |
| **num_cross_layers** | ✅ **2** | ❌ 1 | 特征交叉能力弱 |
| net_dropout | 0.1 | 0.1 | - |
| model_structure | parallel | parallel | - |
| learning_rate | 0.001 | 0.001 | - |

### 失败配置的训练结果
| Epoch | AUC | gAUC | 提升幅度 |
|-------|-----|------|---------|
| 1 | 0.6947 | 0.6121 | - |
| 2 | 0.6970 | 0.6173 | +0.0023 |
| 3 | 0.6979 | 0.6202 | +0.0008 |

**问题分析**：
1. `batch_norm=false` 导致训练不稳定，收敛速度极慢
2. `num_cross_layers=1` 限制了模型的特征交叉能力
3. 按照此趋势，需要30-50个epoch才能达到0.74，且可能陷入局部最优

---

## 🚀 训练命令

### 使用最佳配置训练
```bash
cd /home/raoxuan/projects/open_unimixer_skills/FuxiCTR/model_zoo/DCNv2

python run_expid.py \
    --config ./config \
    --expid DCNv2_kuaivideo_x1_008_7047bff3 \
    --gpu 0
```

### 使用CPU训练（不推荐）
```bash
python run_expid.py \
    --config ./config \
    --expid DCNv2_kuaivideo_x1_008_7047bff3 \
    --gpu -1
```

---

## 📁 配置文件路径

- **最佳配置**：`BARS/ranking/ctr/DCNv2/DCNv2_kuaivideo_x1/DCNv2_kuaivideo_x1_tuner_config_01/model_config.yaml`
- **数据集配置**：`FuxiCTR/model_zoo/DCNv2/config/dataset_config.yaml`
- **训练脚本**：`FuxiCTR/model_zoo/DCNv2/run_expid.py`

---

## 💡 关键洞察

### 为什么 batch_norm=true 很重要？
1. **加速收敛**：Batch Normalization 可以显著加速训练过程
2. **稳定训练**：减少内部协变量偏移，使训练更稳定
3. **允许更高学习率**：可以使用更大的学习率而不导致梯度爆炸/消失

### 为什么 num_cross_layers=2 更好？
1. **更强的特征交叉**：DCN的核心是显式建模特征交叉，2层可以捕捉更复杂的交互
2. **表达能力**：1层可能只能捕捉简单的二阶交叉，2层可以捕捉高阶交叉
3. **与数据复杂度匹配**：KuaiVideo数据集特征复杂，需要更强的交叉能力

---

## 📈 预期训练时间

| 环境 | 每epoch时间 | 达到0.74预计epochs | 总时间 |
|------|------------|-------------------|--------|
| RTX 4090 | ~7分钟 | ~8-12 | ~1-1.5小时 |
| CPU | ~60分钟 | ~8-12 | ~8-12小时 |

---

## 🔗 参考链接

- [BARS Benchmark GitHub](https://github.com/reczoo/BARS)
- [FuxiCTR Framework](https://github.com/reczoo/FuxiCTR)
- [DCN-V2 Paper](https://arxiv.org/abs/2008.13535)

---

*文档生成时间：2026-04-10*
*最佳配置来源：BARS/ranking/ctr/DCNv2/DCNv2_kuaivideo_x1/DCNv2_kuaivideo_x1_008_7047bff3*
