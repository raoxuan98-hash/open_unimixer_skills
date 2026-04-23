#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script for custom models on Frappe_x1 dataset.
Usage:
  python scripts/quick_test_frappe.py --model HeteroAttention
  python scripts/quick_test_frappe.py --model all
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"
DATA_ROOT = PROJECT_ROOT / "data"

# Models to test
MODELS = {
    "HeteroAttention": {
        "class_name": "HeteroAttention",
        "params": {
            "embedding_dim": 40,
            "num_transformer_layers": 2,
            "num_heads": 4,
            "transformer_dropout": 0.1,
            "ffn_dim": 80,
            "use_cls_token": True,
            "use_pos_embedding": True,
            "output_mlp_hidden_units": [80, 40],
            "net_dropout": 0.1,
            "batch_norm": False,
        }
    },
    "RankMixer": {
        "class_name": "RankMixer",
        "params": {
            "embedding_dim": 40,
            "num_transformer_layers": 2,
            "transformer_dropout": 0.1,
            "ffn_dim": 80,
            "use_pos_embedding": True,
            "output_mlp_hidden_units": [80, 40],
            "net_dropout": 0.1,
            "batch_norm": False,
        }
    },
    "HiFormer": {
        "class_name": "HiFormer",
        "params": {
            "embedding_dim": 40,
            "num_transformer_layers": 2,
            "num_heads": 4,
            "low_rank_dim": 8,
            "transformer_dropout": 0.1,
            "ffn_dim": 80,
            "use_pos_embedding": True,
            "output_mlp_hidden_units": [80, 40],
            "net_dropout": 0.1,
            "batch_norm": False,
        }
    },
    "FAT": {
        "class_name": "FAT",
        "params": {
            "embedding_dim": 40,
            "num_transformer_layers": 2,
            "num_heads": 4,
            "basis_num": 4,
            "sinkhorn_iters": 5,
            "transformer_dropout": 0.1,
            "ffn_dim": 80,
            "use_pos_embedding": True,
            "output_mlp_hidden_units": [80, 40],
            "net_dropout": 0.1,
            "batch_norm": False,
        }
    },
    "TokenMixer_Large": {
        "class_name": "TokenMixerLarge",
        "params": {
            "embedding_dim": 40,
            "num_transformer_layers": 2,
            "transformer_dropout": 0.1,
            "ffn_dim": 60,
            "use_pos_embedding": True,
            "output_mlp_hidden_units": [60, 40],
            "net_dropout": 0.1,
            "batch_norm": False,
        }
    },
    "UniMixerLite": {
        "class_name": "UniMixerLite",
        "params": {
            "embedding_dim": 40,
            "num_transformer_layers": 2,
            "block_size": 4,
            "basis_num": 4,
            "sinkhorn_iters": 20,
            "sinkhorn_temperature": 0.2,
            "transformer_dropout": 0.1,
            "ffn_dim": 80,
            "use_pos_embedding": True,
            "output_mlp_hidden_units": [80, 40],
            "net_dropout": 0.1,
            "batch_norm": False,
        }
    },
}

# Common training settings
COMMON_TRAIN_CONFIG = {
    "dataset_id": "frappe_x1",
    "loss": "binary_crossentropy",
    "metrics": ["AUC", "logloss"],
    "task": "binary_classification",
    "optimizer": "adam",
    "learning_rate": 1.0e-3,
    "embedding_regularizer": 0,
    "net_regularizer": 0,
    "batch_size": 2048,
    "epochs": 3,
    "shuffle": True,
    "seed": 2024,
    "monitor": "AUC",
    "monitor_mode": "max",
    "early_stop_patience": 2,
}

BASE_CONFIG = {
    "model_root": "./checkpoints/",
    "num_workers": 3,
    "verbose": 1,
    "early_stop_patience": 2,
    "pickle_feature_encoder": True,
    "save_best_only": True,
    "eval_steps": None,
    "debug_mode": False,
    "group_id": None,
    "use_features": None,
    "feature_specs": None,
    "feature_config": None,
}

DATASET_CONFIG = {
    "frappe_x1": {
        "data_format": "csv",
        "data_root": str(DATA_ROOT),
        "feature_cols": [
            {"active": True, "dtype": "str", "name": "user", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "item", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "daytime", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "weekday", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "isweekend", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "homework", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "cost", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "weather", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "country", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "city", "type": "categorical"},
        ],
        "label_col": {"dtype": "float", "name": "label"},
        "min_categr_count": 1,
        "test_data": str(DATA_ROOT / "Frappe_x1" / "test.csv"),
        "train_data": str(DATA_ROOT / "Frappe_x1" / "train.csv"),
        "valid_data": str(DATA_ROOT / "Frappe_x1" / "valid.csv"),
    }
}


def write_yaml(path, content):
    import yaml
    with open(path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)


def setup_model(model_name, model_info):
    model_dir = FUXICTR_ROOT / "model_zoo" / model_name
    config_dir = model_dir / "config"
    config_dir.mkdir(exist_ok=True)

    # Write dataset_config.yaml (used by --dataset_config override)
    write_yaml(config_dir / "dataset_config.yaml", DATASET_CONFIG)

    # Write model_config.yaml
    exp_id = f"{model_info['class_name']}_frappe_quick"
    model_config = {
        "Base": BASE_CONFIG,
        exp_id: {
            "model": model_info["class_name"],
            **COMMON_TRAIN_CONFIG,
            **model_info["params"],
        }
    }
    write_yaml(config_dir / "model_config.yaml", model_config)

    return exp_id


def run_model(model_name, exp_id, gpu=-1):
    model_dir = FUXICTR_ROOT / "model_zoo" / model_name
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main" / "run_expid.py"),
        "--model", model_name,
        "--expid", exp_id,
        "--dataset_config", str(model_dir / "config"),
        "--gpu", str(gpu),
    ]
    print(f"\n{'='*60}")
    print(f"Running {model_name} | expid={exp_id}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index")
    args = parser.parse_args()

    target_models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in target_models:
        if model_name not in MODELS:
            print(f"[ERROR] Unknown model: {model_name}")
            continue
        info = MODELS[model_name]
        exp_id = setup_model(model_name, info)
        ret = run_model(model_name, exp_id, gpu=args.gpu)
        status = "OK" if ret == 0 else f"FAILED (code={ret})"
        print(f"\n[RESULT] {model_name}: {status}")


if __name__ == "__main__":
    main()
