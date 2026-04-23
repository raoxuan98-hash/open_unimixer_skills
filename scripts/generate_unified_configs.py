#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate unified large-scale experiment configs for all 6 custom models
across 4 datasets with consistent hyperparameter rules.
"""

import os
import sys
import shutil
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"
DATA_ROOT = PROJECT_ROOT / "data"

DATASETS = {
    "movielenslatest_x1": {
        "data_root": str(DATA_ROOT),
        "feature_cols": [
            {"active": True, "dtype": "str", "name": "user_id", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "item_id", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "tag_id", "type": "categorical"},
        ],
        "label_col": {"dtype": "float", "name": "label"},
        "min_categr_count": 1,
        "test_data": str(DATA_ROOT / "MovielensLatest_x1" / "test.csv"),
        "train_data": str(DATA_ROOT / "MovielensLatest_x1" / "train.csv"),
        "valid_data": str(DATA_ROOT / "MovielensLatest_x1" / "valid.csv"),
        "embedding_dim": 40,
        "batch_size": 4096,
        "metrics": ["AUC", "logloss"],
        "monitor": "AUC",
    },
    "frappe_x1": {
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
        "embedding_dim": 56,
        "batch_size": 4096,
        "metrics": ["AUC", "logloss"],
        "monitor": "AUC",
    },
    "kuaivideo_x1": {
        "data_root": str(DATA_ROOT),
        "feature_cols": [
            {"active": True, "dtype": "int", "name": "group_id", "preprocess": "copy_from(user_id)", "remap": False, "type": "meta"},
            {"active": True, "dtype": "str", "name": "user_id", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "item_id", "type": "categorical"},
            {"active": True, "dtype": "str", "embedding_dim": 64, "min_categr_count": 1, "name": "item_emb", "preprocess": "copy_from(item_id)", "pretrained_emb": "/home/raoxuan/projects/open_unimixer_skills/data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5", "type": "categorical"},
            {"active": True, "dtype": "str", "max_len": 100, "name": "pos_items", "padding": "pre", "share_embedding": "item_id", "splitter": "^", "type": "sequence"},
            {"active": True, "dtype": "str", "max_len": 100, "name": "neg_items", "padding": "pre", "share_embedding": "item_id", "splitter": "^", "type": "sequence"},
            {"active": True, "dtype": "str", "embedding_dim": 64, "max_len": 100, "min_categr_count": 1, "name": "pos_items_emb", "padding": "pre", "preprocess": "copy_from(pos_items)", "pretrained_emb": "/home/raoxuan/projects/open_unimixer_skills/data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5", "share_embedding": "item_emb", "splitter": "^", "type": "sequence"},
            {"active": True, "dtype": "str", "embedding_dim": 64, "max_len": 100, "min_categr_count": 1, "name": "neg_items_emb", "padding": "pre", "preprocess": "copy_from(neg_items)", "pretrained_emb": "/home/raoxuan/projects/open_unimixer_skills/data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5", "share_embedding": "item_emb", "splitter": "^", "type": "sequence"},
        ],
        "feature_specs": [
            {"feature_encoder": "nn.Linear(64, 64, bias=False)", "name": "item_emb"},
            {"feature_encoder": "layers.MaskedAveragePooling()", "name": "pos_items"},
            {"feature_encoder": "layers.MaskedAveragePooling()", "name": "neg_items"},
            {"feature_encoder": ["layers.MaskedAveragePooling()", "nn.Linear(64, 64, bias=False)"], "name": "pos_items_emb"},
            {"feature_encoder": ["layers.MaskedAveragePooling()", "nn.Linear(64, 64, bias=False)"], "name": "neg_items_emb"},
        ],
        "label_col": {"dtype": "float", "name": "is_click"},
        "min_categr_count": 10,
        "test_data": str(DATA_ROOT / "KuaiShou" / "KuaiVideo_x1" / "test.csv"),
        "train_data": str(DATA_ROOT / "KuaiShou" / "KuaiVideo_x1" / "train.csv"),
        "valid_data": str(DATA_ROOT / "KuaiShou" / "KuaiVideo_x1" / "test.csv"),
        "group_id": "group_id",
        "embedding_dim": 40,
        "batch_size": 8192,
        "metrics": ["gAUC", "AUC", "logloss"],
        "monitor": {"AUC": 1, "gAUC": 1},
    },
    "taobaoad_x1": {
        "data_root": str(DATA_ROOT),
        "feature_cols": [
            {"active": True, "dtype": "int", "name": "group_id", "preprocess": "copy_from(userid)", "remap": False, "type": "meta"},
            {"active": True, "dtype": "str", "name": ["userid", "cms_segid", "cms_group_id", "final_gender_code", "age_level", "pvalue_level", "shopping_level", "occupation", "new_user_class_level", "adgroup_id", "cate_id", "campaign_id", "customer", "brand", "pid", "btag"], "type": "categorical"},
            {"active": True, "dtype": "float", "name": "price", "type": "numeric"},
            {"active": True, "dtype": "str", "max_len": 50, "name": "cate_his", "padding": "pre", "share_embedding": "cate_id", "splitter": "^", "type": "sequence"},
            {"active": True, "dtype": "str", "max_len": 50, "name": "brand_his", "padding": "pre", "share_embedding": "brand", "splitter": "^", "type": "sequence"},
            {"active": True, "dtype": "str", "max_len": 50, "name": "btag_his", "padding": "pre", "share_embedding": "btag", "splitter": "^", "type": "sequence"},
        ],
        "label_col": {"dtype": "float", "name": "clk"},
        "min_categr_count": 10,
        "test_data": str(DATA_ROOT / "TaobaoAd_x1" / "test.csv"),
        "train_data": str(DATA_ROOT / "TaobaoAd_x1" / "train.csv"),
        "valid_data": str(DATA_ROOT / "TaobaoAd_x1" / "test.csv"),
        "group_id": "group_id",
        "embedding_dim": 48,
        "batch_size": 8192,
        "metrics": ["gAUC", "AUC", "logloss"],
        "monitor": {"AUC": 1, "gAUC": 1},
    },
    "microvideo1.7m_x1": {
        "data_root": str(DATA_ROOT),
        "feature_cols": [
            {"active": True, "dtype": "int", "name": "group_id", "preprocess": "copy_from(user_id)", "remap": False, "type": "meta"},
            {"active": True, "dtype": "str", "name": "user_id", "type": "categorical"},
            {"active": True, "dtype": "str", "embedding_dim": 64, "name": "item_id", "pretrained_emb": "/home/raoxuan/projects/open_unimixer_skills/data/MicroVideo1.7M_x1/item_image_emb_dim64.h5", "type": "categorical"},
            {"active": True, "dtype": "str", "name": "cate_id", "type": "categorical"},
            {"active": True, "dtype": "str", "embedding_dim": 64, "max_len": 100, "name": "clicked_items", "padding": "pre", "pretrained_emb": "/home/raoxuan/projects/open_unimixer_skills/data/MicroVideo1.7M_x1/item_image_emb_dim64.h5", "splitter": "^", "type": "sequence"},
            {"active": True, "dtype": "str", "max_len": 100, "name": "clicked_categories", "padding": "pre", "share_embedding": "cate_id", "splitter": "^", "type": "sequence"},
            {"active": False, "dtype": "str", "name": "timestamp", "type": "categorical"},
        ],
        "label_col": {"dtype": "float", "name": "is_click"},
        "min_categr_count": 1,
        "test_data": str(DATA_ROOT / "MicroVideo1.7M_x1" / "test.csv"),
        "train_data": str(DATA_ROOT / "MicroVideo1.7M_x1" / "train.csv"),
        "valid_data": str(DATA_ROOT / "MicroVideo1.7M_x1" / "test.csv"),
        "embedding_dim": 40,
        "batch_size": 4096,
        "metrics": ["AUC", "logloss"],
        "monitor": "AUC",
    },
}

MODELS = ["HeteroAttention", "RankMixer", "HiFormer", "FAT", "TokenMixer_Large", "UniMixer_lite", "TransformerCTR"]

COMMON_TRAIN = {
    "loss": "binary_crossentropy",
    "task": "binary_classification",
    "optimizer": "adam",
    "learning_rate": 1.0e-3,
    "embedding_regularizer": 0,
    "net_regularizer": 0,
    "epochs": 100,
    "shuffle": True,
    "seed": 2024,
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


def get_model_params(model_name, embedding_dim, num_fields=1):
    # RankMixer requires embedding_dim % num_fields == 0
    if model_name == "RankMixer":
        if embedding_dim % num_fields != 0:
            embedding_dim = ((embedding_dim // num_fields) + 1) * num_fields
            print(f"[INFO] Adjusted RankMixer embedding_dim to {embedding_dim} for {num_fields} fields")

    # FFN rules
    if model_name == "TokenMixer_Large":
        ffn_dim = int(1.5 * embedding_dim)
    else:
        ffn_dim = 2 * embedding_dim
    output_mlp = [ffn_dim, embedding_dim]

    common = {
        "embedding_dim": embedding_dim,
        "num_transformer_layers": 2,
        "transformer_dropout": 0.1,
        "ffn_dim": ffn_dim,
        "use_pos_embedding": True,
        "output_mlp_hidden_units": output_mlp,
        "net_dropout": 0.1,
        "batch_norm": False,
        "data_block_size": 1000000,
    }

    if model_name == "HeteroAttention":
        return {**common, "num_heads": 4, "use_cls_token": True}
    elif model_name == "RankMixer":
        return {**common}
    elif model_name == "HiFormer":
        return {**common, "num_heads": 4, "low_rank_dim": 8}
    elif model_name == "FAT":
        return {**common, "num_heads": 4, "basis_num": 4, "sinkhorn_iters": 20}
    elif model_name == "TokenMixer_Large":
        return {**common}
    elif model_name == "UniMixer_lite":
        return {
            **common,
            "block_size": 4,
            "basis_num": 4,
            "sinkhorn_iters": 20,
            "sinkhorn_temperature": 0.2,
        }
    elif model_name == "TransformerCTR":
        return {
            **common,
            "num_heads": 4,
            "use_cls_token": True,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_class_name(model_name):
    mapping = {
        "HeteroAttention": "HeteroAttention",
        "RankMixer": "RankMixer",
        "HiFormer": "HiFormer",
        "FAT": "FAT",
        "TokenMixer_Large": "TokenMixerLarge",
        "UniMixer_lite": "UniMixerLite",
        "TransformerCTR": "TransformerCTR",
    }
    return mapping[model_name]


def generate_configs():
    for model_name in MODELS:
        model_dir = FUXICTR_ROOT / "model_zoo" / model_name
        config_dir = model_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # Copy run_expid.py and fuxictr_version.py if missing
        run_script = model_dir / "run_expid.py"
        if not run_script.exists():
            src_script = FUXICTR_ROOT / "model_zoo" / "DCNv2" / "run_expid.py"
            shutil.copy(src_script, run_script)
            print(f"[INFO] Copied run_expid.py to {model_name}")
        version_file = model_dir / "fuxictr_version.py"
        if not version_file.exists():
            src_version = FUXICTR_ROOT / "model_zoo" / "DCNv2" / "fuxictr_version.py"
            shutil.copy(src_version, version_file)
            print(f"[INFO] Copied fuxictr_version.py to {model_name}")

        dataset_configs = {}
        model_configs = {"Base": BASE_CONFIG}

        for dataset_id, ds_info in DATASETS.items():
            embedding_dim = ds_info["embedding_dim"]
            batch_size = ds_info["batch_size"]
            metrics = ds_info["metrics"]
            monitor = ds_info["monitor"]

            # Dataset config
            dataset_config = {
                "data_format": "csv",
                "data_root": ds_info["data_root"],
                "feature_cols": ds_info["feature_cols"],
                "label_col": ds_info["label_col"],
                "min_categr_count": ds_info["min_categr_count"],
                "test_data": ds_info["test_data"],
                "train_data": ds_info["train_data"],
                "valid_data": ds_info["valid_data"],
            }
            if "feature_specs" in ds_info:
                dataset_config["feature_specs"] = ds_info["feature_specs"]
            dataset_configs[dataset_id] = dataset_config

            # Compute num_fields for RankMixer compatibility
            num_fields = sum(
                1 if isinstance(col.get("name"), str) else len(col.get("name", []))
                for col in ds_info["feature_cols"]
                if col.get("active") != False and col.get("type") != "meta"
            )

            # Model config
            exp_id = f"{get_class_name(model_name)}_{dataset_id}_unified"
            model_param = get_model_params(model_name, embedding_dim, num_fields)
            train_cfg = {
                **COMMON_TRAIN,
                "dataset_id": dataset_id,
                "batch_size": batch_size,
                "metrics": metrics,
                "monitor": monitor,
                "model": get_class_name(model_name),
            }
            if "group_id" in ds_info:
                train_cfg["group_id"] = ds_info["group_id"]

            model_configs[exp_id] = {**train_cfg, **model_param}

        # Write dataset_config.yaml
        with open(config_dir / "dataset_config.yaml", "w") as f:
            yaml.dump(dataset_configs, f, default_flow_style=False, sort_keys=False)

        # Write model_config.yaml
        with open(config_dir / "model_config.yaml", "w") as f:
            yaml.dump(model_configs, f, default_flow_style=False, sort_keys=False)

        print(f"[DONE] {model_name}: {len(DATASETS)} datasets configured")


if __name__ == "__main__":
    generate_configs()
    print("\nAll unified configs generated successfully.")
