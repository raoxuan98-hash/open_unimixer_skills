# Open Unimixer Skills

用于 CTR（Click-Through Rate）模型评估的 Kimi CLI Skills 集合。

## 简介

本项目提供一系列 skills，帮助用户从零开始搭建 BARS + FuxiCTR 环境，下载数据集，并训练各种 CTR 预测模型。

## Skills 列表

| Skill | 路径 | 功能 |
|-------|------|------|
| **project-setup** | `skills/project-setup/` | 从 GitHub 克隆项目、安装依赖、设置数据目录关联 |
| **dataset-download** | `skills/dataset-download/` | 下载和准备 CTR 评估数据集 |
| **ctr-training** | `skills/ctr-training/` | 配置和训练各种 CTR 模型（DeepFM、DCN、DIN 等） |

## 快速开始

### 1. 项目初始化

参考 `skills/project-setup/SKILL.md`：

```bash
# 克隆项目仓库
git clone https://github.com/reczoo/BARS.git
git clone https://github.com/reczoo/FuxiCTR.git

# 安装依赖
pip install fuxictr>=2.3.7

# 设置数据目录软链接
ln -sf $(pwd)/data FuxiCTR/data
```

### 2. 下载数据集

参考 `skills/dataset-download/SKILL.md`：

```bash
cd data
git clone https://huggingface.co/datasets/reczoo/MovielensLatest_x1
```

### 3. 训练模型

参考 `skills/ctr-training/SKILL.md`：

```bash
cd FuxiCTR/model_zoo/DCNv2
python run_expid.py --config ./config --expid MyExp --gpu -1
```

## 支持的数据集

- MovielensLatest_x1
- Frappe_x1
- TaobaoAd_x1
- KuaiVideo_x1
- Criteo_x1

## 支持的模型

- DeepFM
- DCN / DCNv2
- DIN
- xDeepFM
- AutoInt
- NFM
- AFM
- 更多...

## 大规模实验（6 模型 × 4 数据集）

本项目提供了统一配置生成和批量运行脚本，支持在 4 个数据集（MovielensLatest、Frappe、KuaiVideo、TaobaoAd）上同时评估 6 个自定义模型（HeteroAttention、RankMixer、HiFormer、FAT、TokenMixerLarge、UniMixerLite）。

```bash
# 生成统一实验配置
python scripts/generate_unified_configs.py

# 预览所有实验命令（不实际运行）
python scripts/run_all_unified.py --dry-run

# 启动全部实验
python scripts/run_all_unified.py --gpu 0

# 只运行特定模型或数据集
python scripts/run_all_unified.py --model UniMixer_lite --dataset frappe_x1 --gpu 0
```

## 相关项目

- [FuxiCTR](https://github.com/reczoo/FuxiCTR) - CTR 预测深度学习框架
- [BARS](https://github.com/reczoo/BARS) - CTR 模型基准测试
- [BARS Datasets](https://github.com/reczoo/Datasets) - 公开数据集集合

## License

Apache License 2.0
