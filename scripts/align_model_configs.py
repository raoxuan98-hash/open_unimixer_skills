#!/usr/bin/env python3
"""
Batch update model_config.yaml for all target models to align hyperparameters
fairly across datasets.
"""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR" / "model_zoo"

MODELS = [
    "FAT",
    "HiFormer",
    "TransformerCTR",
    "HeteroAttention",
    "RankMixer",
    "TokenMixer_Large",
    "UniMixer_lite",
    "HybridMixer",
]

# Alignment config per dataset (size-based grouping)
UNIFIED_ALIGNMENT = {
    "frappe_x1": {
        "embedding_dim": 128,
        "ffn_in_dim": None,
        "ffn_out_dim": 128,
        "net_dropout": 0.2,
        "embedding_regularizer": 0.005,
        "net_regularizer": 5.0e-05,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },
    "movielenslatest_x1": {
        "embedding_dim": 128,
        "ffn_in_dim": None,
        "ffn_out_dim": 128,
        "net_dropout": 0.2,
        "embedding_regularizer": 0.005,
        "net_regularizer": 5.0e-05,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },
    "kuaivideo_x1": {
        "embedding_dim": 128,
        "ffn_in_dim": None,
        "ffn_out_dim": 512,
        "net_dropout": 0.2,
        "embedding_regularizer": 0.005,
        "net_regularizer": 5.0e-05,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },
    "taobaoad_x1": {
        "embedding_dim": 128,
        "ffn_in_dim": None,
        "ffn_out_dim": 768,
        "net_dropout": 0.1,
        "embedding_regularizer": 0.001,
        "net_regularizer": 1.0e-05,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },
    "microvideo1.7m_x1": {
        "embedding_dim": 128,
        "ffn_in_dim": 128,
        "ffn_out_dim": 768,
        "net_dropout": 0.1,
        "embedding_regularizer": 0.001,
        "net_regularizer": 1.0e-05,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },
}


def update_config(model_name):
    config_path = FUXICTR_ROOT / model_name / "config" / "model_config.yaml"
    if not config_path.exists():
        print(f"[SKIP] Config not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    updated_count = 0
    for section_name, section in config.items():
        if not isinstance(section, dict):
            continue
        dataset_id = section.get("dataset_id")
        if dataset_id not in UNIFIED_ALIGNMENT:
            continue

        align = UNIFIED_ALIGNMENT[dataset_id]
        # Remove legacy ffn_dim if present; replace with ffn_out_dim
        if "ffn_dim" in section:
            del section["ffn_dim"]
        for key, value in align.items():
            old_value = section.get(key)
            section[key] = value
            if old_value != value:
                updated_count += 1
                print(f"  [{model_name}/{section_name}] {key}: {old_value} -> {value}")

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"[DONE] {model_name}: {updated_count} fields updated.")


def main():
    for model in MODELS:
        update_config(model)
    print("\nAll model configs aligned.")


if __name__ == "__main__":
    main()
