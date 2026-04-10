---
name: dataset-download
description: 下载和准备 CTR 模型评估所需的数据集（MovielensLatest、Frappe、KuaiVideo 等）
---

# 数据集下载 Skill

本 skill 提供下载和准备 BARS/FuxiCTR 基准测试数据集的方法。

## 支持的数据集

| 数据集 | 大小 | 样本数 | 特征类型 |
|--------|------|--------|----------|
| MovielensLatest_x1 | ~16MB | 200万+ | 用户、物品、标签 |
| Frappe_x1 | ~3MB | 28万+ | 上下文特征（时间、地点等） |
| TaobaoAd_x1 | ~2.3GB | 1.1亿+ | 广告展示/点击、用户画像 |
| KuaiVideo_x1 | ~2.2GB | 1300万+ | 用户行为序列 + 视觉嵌入 |
| Criteo_x1 | ~450MB | 4500万+ | 广告点击率 |

## 使用方法

### 方法1：Git LFS 克隆（推荐）

```bash
# 进入数据目录
mkdir -p data && cd data

# 克隆数据集仓库（包含完整版本控制）
git clone https://huggingface.co/datasets/reczoo/MovielensLatest_x1
git clone https://huggingface.co/datasets/reczoo/Frappe_x1
git clone https://huggingface.co/datasets/reczoo/TaobaoAd_x1
git clone https://huggingface.co/datasets/reczoo/KuaiVideo_x1

# 确保安装了 git-lfs
git lfs pull
```

### 方法2：直接下载 ZIP 文件

```bash
cd data

# MovielensLatest_x1
curl -L -o MovielensLatest_x1.zip \
  "https://huggingface.co/datasets/reczoo/MovielensLatest_x1/resolve/main/MovielensLatest_x1.zip"
unzip -q MovielensLatest_x1.zip -d MovielensLatest_x1

# Frappe_x1
curl -L -o Frappe_x1.zip \
  "https://huggingface.co/datasets/reczoo/Frappe_x1/resolve/main/Frappe_x1.zip"
unzip -q Frappe_x1.zip -d Frappe_x1

# TaobaoAd_x1（较大，约2.3GB）
curl -L -o TaobaoAd_x1.zip \
  "https://huggingface.co/datasets/reczoo/TaobaoAd_x1/resolve/main/TaobaoAd_x1.zip?download=true"
unzip -q TaobaoAd_x1.zip -d TaobaoAd_x1

# KuaiVideo_x1（较大）
curl -L -o KuaiVideo_x1.zip \
  "https://huggingface.co/datasets/reczoo/KuaiVideo_x1/resolve/main/KuaiVideo_x1.zip?download=true"
unzip -q KuaiVideo_x1.zip -d KuaiVideo_x1
```

### 方法3：Python 脚本下载

```python
from huggingface_hub import hf_hub_download
import os

def download_dataset(repo_id, local_dir, files=None):
    """下载数据集
    
    Args:
        repo_id: HuggingFace 仓库ID，如 "reczoo/MovielensLatest_x1"
        local_dir: 本地保存目录
        files: 指定文件列表，None则下载所有
    """
    os.makedirs(local_dir, exist_ok=True)
    
    if files is None:
        # 默认下载 train/valid/test
        files = ["train.csv", "valid.csv", "test.csv"]
    
    for filename in files:
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

# 使用示例
download_dataset("reczoo/MovielensLatest_x1", "data/MovielensLatest_x1")
download_dataset("reczoo/Frappe_x1", "data/Frappe_x1")
download_dataset("reczoo/TaobaoAd_x1", "data/TaobaoAd_x1")
```

## 数据集结构

下载完成后，每个数据集应包含：

```
data/
├── MovielensLatest_x1/
│   ├── train.csv      # 训练集
│   ├── valid.csv      # 验证集
│   └── test.csv       # 测试集
├── Frappe_x1/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── TaobaoAd_x1/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── KuaiVideo_x1/
    ├── train.csv
    ├── test.csv
    └── item_visual_emb_dim64.h5  # 预训练视觉嵌入
```

## 验证下载

```bash
# 检查文件完整性
ls -la data/MovielensLatest_x1/

# 查看样本数量
wc -l data/MovielensLatest_x1/*.csv

# 查看数据格式（列名）
head -1 data/MovielensLatest_x1/train.csv
```

## 常见问题

### Q: 下载速度慢怎么办？
A: 
- 使用镜像源或代理
- 对于大文件（如 KuaiVideo），建议直接使用浏览器下载后上传
- 使用 `wget` 或 `aria2` 等多线程下载工具

### Q: 数据集文件损坏？
A: 检查 MD5 校验值：
```bash
md5sum data/KuaiVideo_x1/train.csv
# 应与官方公布的校验值对比
```

### Q: huggingface-cli 命令找不到？
A: 需要安装 huggingface_hub：
```bash
pip install huggingface-hub
```

## 相关资源

- [HuggingFace 数据集主页](https://huggingface.co/reczoo)
- [BARS Datasets GitHub](https://github.com/reczoo/Datasets)
- [数据集统计信息](https://github.com/reczoo/BARS/tree/main/datasets)
