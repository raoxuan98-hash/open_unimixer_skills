---
name: project-setup
description: 从零开始搭建 BARS + FuxiCTR CTR 模型评估项目环境
---

# 项目初始化 Setup Skill

本 skill 指导如何从 GitHub 从零开始下载、配置和运行 BARS + FuxiCTR 项目。

## 项目架构

```
项目根目录/
├── BARS/                       # 基准测试框架（配置和实验记录）
│   └── ranking/ctr/            # CTR 模型配置
│       ├── DCNv2/
│       ├── DIN/
│       ├── DeepFM/
│       └── ...
├── FuxiCTR/                    # 深度学习框架（模型实现）
│   ├── model_zoo/              # 各种模型实现
│   │   ├── DCNv2/
│   │   ├── DIN/
│   │   ├── DeepFM/
│   │   └── ...
│   └── data/                   # 数据目录（需要创建软链接）
└── data/                       # 你的数据集目录
    ├── MovielensLatest_x1/
    ├── Frappe_x1/
    └── ...
```

## 步骤1：克隆项目仓库

```bash
# 创建项目目录
mkdir -p ctr_benchmark
cd ctr_benchmark

# 克隆 BARS 项目（包含数据集配置和实验记录）
git clone https://github.com/reczoo/BARS.git

# 克隆 FuxiCTR 项目（包含模型实现）
git clone https://github.com/reczoo/FuxiCTR.git
```

## 步骤2：安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活环境
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装 FuxiCTR 框架
pip install fuxictr>=2.3.7

# 验证安装
python -c "import fuxictr; print(f'FuxiCTR version: {fuxictr.__version__}')"
```

## 步骤3：设置数据目录关联

FuxiCTR 默认从 `FuxiCTR/data/` 读取数据，需要创建软链接指向你的数据目录：

```bash
# 在项目根目录执行
# 方法1：如果数据目录在项目根目录
ln -sf $(pwd)/data FuxiCTR/data

# 方法2：如果数据目录在其他位置（使用绝对路径）
ln -sf /absolute/path/to/your/data FuxiCTR/data
```

验证软链接：
```bash
ls -la FuxiCTR/data
# 应该显示指向你的数据目录的软链接
```

## 步骤4：修复 NumPy 2.0 兼容性（可选但推荐）

```bash
# 修复 fuxictr 中的 np.Inf 问题
sed -i 's/np.Inf/np.inf/g' \
  venv/lib/python*/site-packages/fuxictr/pytorch/models/rank_model.py
```

## 步骤5：下载数据集

参考 `dataset-download` skill：

```bash
# 创建数据目录
mkdir -p data
cd data

# 下载数据集（示例：MovieLens）
git clone https://huggingface.co/datasets/reczoo/MovielensLatest_x1

# 返回项目根目录
cd ..
```

## 步骤6：配置并运行模型

### 6.1 复制 BARS 配置到 FuxiCTR

```bash
# 示例：配置 DCNv2 on MovielensLatest_x1
# 复制数据集配置
cp BARS/ranking/ctr/DCNv2/DCNv2_movielenslatest_x1_*/dataset_config.yaml \
   FuxiCTR/model_zoo/DCNv2/config/dataset_config.yaml

# 复制模型配置
cp BARS/ranking/ctr/DCNv2/DCNv2_movielenslatest_x1_*/model_config.yaml \
   FuxiCTR/model_zoo/DCNv2/config/model_config.yaml
```

### 6.2 修改数据路径

编辑 `FuxiCTR/model_zoo/DCNv2/config/dataset_config.yaml`，
将数据路径改为**绝对路径**：

```yaml
movielenslatest_x1:
    data_root: /absolute/path/to/data/
    train_data: /absolute/path/to/data/MovielensLatest_x1/train.csv
    valid_data: /absolute/path/to/data/MovielensLatest_x1/valid.csv
    test_data: /absolute/path/to/data/MovielensLatest_x1/test.csv
    # ... 其他配置
```

### 6.3 运行训练

```bash
cd FuxiCTR/model_zoo/DCNv2

# CPU 训练
python run_expid.py \
    --config ./config \
    --expid DCNv2_movielenslatest_x1_001 \
    --gpu -1

# GPU 训练
python run_expid.py \
    --config ./config \
    --expid DCNv2_movielenslatest_x1_001 \
    --gpu 0
```

## 一键初始化脚本

```bash
#!/bin/bash
# setup.sh - 一键初始化项目

set -e

echo "=== Step 1: Clone repositories ==="
git clone https://github.com/reczoo/BARS.git || echo "BARS already exists"
git clone https://github.com/reczoo/FuxiCTR.git || echo "FuxiCTR already exists"

echo "=== Step 2: Install dependencies ==="
pip install fuxictr>=2.3.7

echo "=== Step 3: Setup data directory ==="
mkdir -p data
ln -sf $(pwd)/data FuxiCTR/data

echo "=== Step 4: Fix NumPy compatibility ==="
FUXICTR_PATH=$(python -c "import fuxictr; import os; print(os.path.dirname(fuxictr.__file__))" 2>/dev/null || echo "")
if [ -n "$FUXICTR_PATH" ]; then
    sed -i 's/np.Inf/np.inf/g' "$FUXICTR_PATH/pytorch/models/rank_model.py" 2>/dev/null || true
fi

echo "=== Setup complete! ==="
echo "Next steps:"
echo "1. Download datasets: cd data && git clone https://huggingface.co/datasets/reczoo/MovielensLatest_x1"
echo "2. Configure model: edit FuxiCTR/model_zoo/DCNv2/config/*.yaml"
echo "3. Run training: cd FuxiCTR/model_zoo/DCNv2 && python run_expid.py --config ./config --expid EXP --gpu -1"
```

## 验证安装

```bash
# 检查项目结构
ls -la
# 预期输出：BARS/  FuxiCTR/  data/  venv/

# 检查 FuxiCTR 版本
python -c "import fuxictr; print(fuxictr.__version__)"
# 预期输出：2.3.7 或更高

# 检查数据软链接
ls -la FuxiCTR/data
# 预期输出：指向你的数据目录的软链接

# 测试导入模型
python -c "cd FuxiCTR/model_zoo/DCNv2 && import src; print('Model import OK')"
```

## 常见问题

### Q: pip install fuxictr 失败？
A: 尝试使用镜像源或升级 pip：
```bash
pip install --upgrade pip
pip install fuxictr>=2.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 软链接创建失败？
A: Windows 用户需要使用管理员权限或改用复制：
```bash
# Windows 替代方案（复制而非软链接）
xcopy /E /I data FuxiCTR\data
```

### Q: 找不到 run_expid.py？
A: 确保在正确的模型目录下：
```bash
cd FuxiCTR/model_zoo/DCNv2  # 或其他模型目录
ls run_expid.py  # 确认文件存在
```

### Q: BARS 和 FuxiCTR 的版本不匹配？
A: 参考 BARS 中的 environments.txt 安装对应版本：
```bash
cat BARS/ranking/ctr/DCNv2/DCNv2_*/environments.txt
```

## 相关 Skills

- `dataset-download` - 下载数据集
- `ctr-training` - 配置和训练 CTR 模型

## 参考资源

- [FuxiCTR GitHub](https://github.com/reczoo/FuxiCTR)
- [BARS GitHub](https://github.com/reczoo/BARS)
- [官方文档](https://fuxictr.readthedocs.io/)
