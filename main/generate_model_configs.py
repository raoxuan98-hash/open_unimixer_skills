import os
import sys
import yaml
from pathlib import Path
from ruamel.yaml import YAML

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"
GLOBAL_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"

# Target models and their class names (from main/run_expid.py)
MODEL_CLASS_MAP = {
    "DCNv2": "DCNv2",
    "WuKong": "WuKong",
    "FAT": "FAT",
    "HiFormer": "HiFormer",
    "TransformerCTR": "TransformerCTR",
    "HeteroAttention": "HeteroAttention",
    "RankMixer": "RankMixer",
    "TokenMixer_Large": "TokenMixerLarge",
    "UniMixer_lite": "UniMixerLite",
}

def generate_configs():
    # Initialize YAML parser
    yaml_parser = YAML()
    yaml_parser.preserve_quotes = True
    yaml_parser.indent(mapping=2, sequence=4, offset=2)

    # 1. Load global dataset config
    with open(GLOBAL_DATASET_CONFIG, 'r') as f:
        global_dataset_dict = yaml_parser.load(f)
    
    datasets = list(global_dataset_dict.keys())
    print(f"Found datasets: {datasets}")

    for model_name, class_name in MODEL_CLASS_MAP.items():
        print(f"\nProcessing model: {model_name}")
        model_dir = FUXICTR_ROOT / "model_zoo" / model_name
        config_dir = model_dir / "config"
        
        if not model_dir.exists():
            print(f"  Warning: Model directory {model_dir} not found. Skipping.")
            continue
            
        config_dir.mkdir(parents=True, exist_ok=True)

        # 2. Generate/Update dataset_config.yaml
        model_dataset_config_path = config_dir / "dataset_config.yaml"
        # We just overwrite it with the global one to ensure consistency
        with open(model_dataset_config_path, 'w') as f:
            yaml_parser.dump(global_dataset_dict, f)
        print(f"  Updated {model_dataset_config_path}")

        # 3. Generate/Update model_config.yaml
        model_config_path = config_dir / "model_config.yaml"
        model_config_dict = None
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_config_dict = yaml_parser.load(f)
        
        if not model_config_dict:
            # Create a basic model config if it doesn't exist
            model_config_dict = {
                "Base": {
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
                    "feature_config": None
                }
            }

        # Ensure all datasets have an experiment ID
        for dataset_id in datasets:
            # Use class_name for experiment ID prefix to match existing conventions
            exp_id = f"{class_name}_{dataset_id}_unified"
            
            # If the experiment already exists, we might want to preserve it
            # but ensure at least the basic fields are correct.
            if exp_id not in model_config_dict:
                # Use a default template for new experiments
                # Try to copy from an existing experiment of the same model if available
                existing_exps = [k for k in model_config_dict.keys() if k != "Base"]
                if existing_exps:
                    # Copy from the first existing experiment as a template
                    new_exp = dict(model_config_dict[existing_exps[0]])
                    new_exp["dataset_id"] = dataset_id
                    # Update metrics if it's kuaivideo or taobaoad which often use gAUC
                    if "kuaivideo" in dataset_id or "taobaoad" in dataset_id:
                        new_exp["metrics"] = ["gAUC", "AUC", "logloss"]
                        new_exp["monitor"] = {"AUC": 1, "gAUC": 1}
                    else:
                        new_exp["metrics"] = ["AUC", "logloss"]
                        new_exp["monitor"] = "AUC"
                else:
                    # Standard default template
                    new_exp = {
                        "model": class_name,
                        "dataset_id": dataset_id,
                        "loss": "binary_crossentropy",
                        "metrics": ["AUC", "logloss"],
                        "task": "binary_classification",
                        "optimizer": "adam",
                        "learning_rate": 0.001,
                        "embedding_regularizer": 0,
                        "net_regularizer": 0,
                        "batch_size": 4096,
                        "embedding_dim": 40,
                        "epochs": 100,
                        "shuffle": True,
                        "seed": 2024,
                        "monitor": "AUC",
                        "monitor_mode": "max",
                    }
                    if "kuaivideo" in dataset_id or "taobaoad" in dataset_id:
                        new_exp["metrics"] = ["gAUC", "AUC", "logloss"]
                        new_exp["monitor"] = {"AUC": 1, "gAUC": 1}
                        new_exp["group_id"] = "group_id"
                
                model_config_dict[exp_id] = new_exp
                print(f"  Added experiment: {exp_id}")
            else:
                # Experiment exists, just ensure dataset_id is correct
                model_config_dict[exp_id]["dataset_id"] = dataset_id
                print(f"  Verified experiment: {exp_id}")

        # Cleanup redundant experiment IDs that might have been created with model_name prefix
        if model_name != class_name:
            redundant_prefix = f"{model_name}_"
            keys_to_delete = [k for k in model_config_dict.keys() if k.startswith(redundant_prefix)]
            for k in keys_to_delete:
                print(f"  Removing redundant experiment: {k}")
                del model_config_dict[k]

        with open(model_config_path, 'w') as f:
            yaml_parser.dump(model_config_dict, f)
        print(f"  Updated {model_config_path}")

if __name__ == "__main__":
    generate_configs()
