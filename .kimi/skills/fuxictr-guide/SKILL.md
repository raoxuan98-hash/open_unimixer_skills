---
name: fuxictr-guide
description: FuxiCTR framework usage guide for DCN/DIN/DDeepFM models
---

## FuxiCTR Model Guide

This skill provides guidance for using FuxiCTR framework.

### Model Selection

| Model | Use Case | Sequence Support |
|-------|----------|------------------|
| DCN | General CTR | Basic |
| DIN | User interest modeling | Attention-based |
| DeepFM | Feature interactions | Basic |
| DCNv2 | Improved cross network | Basic |

### Key Concepts

1. **Data Preprocessing**: CSV → Parquet + feature_map.json
2. **Feature Types**: categorical, numeric, sequence
3. **Model Training**: Use run_expid.py or custom scripts

### Common Commands

```bash
# Run a model
python run_expid.py --expid DCN_test --gpu 0

# Data preprocessing only
python example1_build_dataset_to_parquet.py
```
