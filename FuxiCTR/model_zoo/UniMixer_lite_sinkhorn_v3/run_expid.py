#!/usr/bin/env python3
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""
Model-specific training entry (auto-detects model name from directory).
Usage:
    python run_expid.py --dataset frappe_x1 --gpu 0
    python run_expid.py --expid RankMixer_frappe_x1_unified --gpu 0
    python run_expid.py --dataset frappe_x1 --epochs 20 --batch_size 2048
"""
import numpy as np
import os
import sys
import logging
import gc
import argparse
from datetime import datetime
from pathlib import Path

# Determine paths from current file location
MODEL_DIR = Path(__file__).parent.resolve()
MODEL_ZOO_DIR = MODEL_DIR.parent
FUXICTR_ROOT = MODEL_ZOO_DIR.parent
PROJECT_ROOT = FUXICTR_ROOT.parent

# Auto-detect model name from directory name
MODEL_NAME = MODEL_DIR.name
MODEL_CLASS_MAP = {
    "DCNv2": "DCNv2",
    "Wukong": "WuKong",
    "FAT": "FAT",
    "HiFormer": "HiFormer",
    "TransformerCTR": "TransformerCTR",
    "HeteroAttention": "HeteroAttention",
    "RankMixer": "RankMixer",
    "TokenMixer_Large": "TokenMixerLarge",
    "UniMixer_lite": "UniMixerLite",
    "UniMixer_lite_sinkhorn": "UniMixerLite",
    "UniMixer_lite_sinkhorn_v2": "UniMixerLite",
}

if MODEL_NAME not in MODEL_CLASS_MAP:
    raise ValueError(f"Unknown model directory: {MODEL_NAME}. Please add it to MODEL_CLASS_MAP.")

sys.path.insert(0, str(FUXICTR_ROOT))

from fuxictr.utils import load_model_config, load_dataset_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything


def import_model_src(model_name):
    """Dynamically import the src package from a specific model directory."""
    model_dir = MODEL_ZOO_DIR / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Clean up any previously imported 'src' modules to avoid cross-model pollution
    for mod_name in list(sys.modules.keys()):
        if mod_name == "src" or mod_name.startswith("src."):
            del sys.modules[mod_name]

    sys.path.insert(0, str(model_dir))
    try:
        import src
        return src
    except ImportError as e:
        raise ImportError(f"Failed to import src for model {model_name}: {e}")


def run_experiment(model_name, experiment_id, gpu=-1,
                   epochs=None, batch_size=None, net_dropout=None,
                   transformer_dropout=None, attention_dropout=None,
                   embedding_dropout=None, learning_rate=None,
                   embedding_dim=None, ffn_out_dim=None, seed=None,
                   embedding_regularizer=None, net_regularizer=None,
                   num_workers=None, metrics=None, use_cosine_lr=None,
                   double_tokens=None, early_stop_patience=None,
                   reduce_lr_patience=None):
    model_config_dir = MODEL_DIR / "config"

    # 1. Load model config
    model_params = load_model_config(str(model_config_dir), experiment_id)
    dataset_id = model_params["dataset_id"]

    # 2. Load dataset config from the model's own config dir
    dataset_config_dir = str(model_config_dir)
    if not Path(dataset_config_dir, "dataset_config.yaml").exists():
        raise FileNotFoundError(
            f"Dataset config not found at {Path(dataset_config_dir) / 'dataset_config.yaml'}."
        )
    data_params = load_dataset_config(dataset_config_dir, dataset_id)

    # Merge: model_params override data_params
    params = dict(data_params, **model_params)
    params["gpu"] = gpu
    params.setdefault("data_format", "csv")

    # ---- Apply command-line overrides (highest priority) ----
    # Only override if the user explicitly provided the argument.
    cli_overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "net_dropout": net_dropout,
        "transformer_dropout": transformer_dropout,
        "attention_dropout": attention_dropout,
        "embedding_dropout": embedding_dropout,
        "learning_rate": learning_rate,
        "embedding_dim": embedding_dim,
        "ffn_out_dim": ffn_out_dim,
        "seed": seed,
        "embedding_regularizer": embedding_regularizer,
        "net_regularizer": net_regularizer,
        "num_workers": num_workers,
        "metrics": metrics,
        "use_cosine_lr": use_cosine_lr,
        "double_tokens": double_tokens,
        "early_stop_patience": early_stop_patience,
        "reduce_lr_patience": reduce_lr_patience,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            params[key] = value

    # 3. Dataset preparation check
    data_dir = os.path.join(params["data_root"], dataset_id)
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    if params["data_format"] == "csv":
        if not os.path.exists(feature_map_json):
            raise RuntimeError(
                f"Dataset {dataset_id} is not prepared. "
                f"Please run: python prepare_train_data.py --dataset {dataset_id}"
            )
        # Use preprocessed block data paths
        params["train_data"] = os.path.join(data_dir, "train")
        params["valid_data"] = os.path.join(data_dir, "valid")
        test_data_dir = os.path.join(data_dir, "test")
        params["test_data"] = test_data_dir if os.path.exists(test_data_dir) else None

    # 4. Logger & seed
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params.get("seed", 2024))

    # 5. Load FeatureMap
    feature_map = FeatureMap(dataset_id, data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    # 6. Build model
    model_class_name = MODEL_CLASS_MAP[model_name]
    src = import_model_src(model_name)
    model_class = getattr(src, model_class_name)
    model = model_class(feature_map, **params)
    model.count_parameters()

    # Optional: Cosine Annealing LR scheduler
    if params.get("use_cosine_lr", False):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        T_max = params.get("epochs", 100)
        eta_min = params["learning_rate"] / 20.0
        model.scheduler = CosineAnnealingLR(model.optimizer, T_max=T_max, eta_min=eta_min)
        logging.info(f"CosineAnnealingLR scheduler enabled: T_max={T_max}, eta_min={eta_min:.6f}")

    # 7. Train
    params["streaming"] = True
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()

    if model_name in ("UniMixer_lite", "UniMixer_lite_sinkhorn", "UniMixer_lite_sinkhorn_v2"):
        # Custom training strategy for UniMixerLite
        epochs = params.get("epochs", 20)
        model.valid_gen = valid_gen
        model._max_gradient_norm = params.get("max_gradient_norm", 10.)
        model._best_metric = np.inf if model._monitor_mode == "min" else -np.inf
        model._stopping_steps = 0
        model._early_stop_patience = params.get("early_stop_patience") or 2
        model._reduce_lr_patience = params.get("reduce_lr_patience") or 1
        model._steps_per_epoch = len(train_gen)
        model._stop_training = False
        model._total_steps = 0
        model._batch_index = 0
        model._epoch_index = 0
        if model._eval_steps is None:
            model._eval_steps = model._steps_per_epoch

        logging.info("Start training with UniMixerLite strategy: {} batches/epoch".format(model._steps_per_epoch))
        for epoch in range(epochs):
            model._epoch_index = epoch
            temp = 1.0 if epoch < int(epochs * 0.6) else 0.05
            if hasattr(model, 'set_sinkhorn_temperature'):
                model.set_sinkhorn_temperature(temp)
                logging.info(f"Epoch {epoch+1}/{epochs}: Setting sinkhorn_temperature to {temp}")

            logging.info("************ Epoch={} start ************".format(epoch + 1))
            model.train_epoch(train_gen)
            if hasattr(model, 'scheduler') and model.scheduler is not None:
                model.scheduler.step()
                current_lr = model.optimizer.param_groups[0]['lr']
                logging.info("Scheduler step: lr={:.6f}".format(current_lr))
            if model._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(model._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(model.checkpoint))
        model.load_weights(model.checkpoint)
    else:
        model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    test_result = {}
    if params.get("test_data"):
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)

    # 8. Save results to project-level results dir
    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    result_filename = result_dir / f"{model_name}_{dataset_id}_results.csv"
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n'
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'),
                    ' '.join(sys.argv), experiment_id, dataset_id,
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))
    logging.info(f"Results saved to {result_filename}")

    return 0


def main():
    parser = argparse.ArgumentParser(description=f"Training entry for {MODEL_NAME}")
    parser.add_argument('--expid', type=str, default=None, help='Experiment id to run')
    AVAILABLE_DATASETS = ["frappe_x1", "movielenslatest_x1", "kuaivideo_x1", "taobaoad_x1", "microvideo1.7m_x1"]
    parser.add_argument('--dataset', type=str, default=None, choices=AVAILABLE_DATASETS,
                        help=f"Dataset id. Choices: {', '.join(AVAILABLE_DATASETS)}")
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu (default: 0)')

    # Use argparse.SUPPRESS so that missing CLI args do NOT override config file values.
    # Common defaults are shown in help text for reference.
    parser.add_argument('--num_workers', type=int, default=argparse.SUPPRESS,
                        help='Number of workers for data loading (common: 8)')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS,
                        help='Number of training epochs (common: 16)')
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        help='Batch size (common: 4096)')
    parser.add_argument('--net_dropout', type=float, default=argparse.SUPPRESS,
                        help='Net dropout rate (common: 0.0)')
    parser.add_argument('--transformer_dropout', type=float, default=argparse.SUPPRESS,
                        help='Transformer dropout rate (common: 0.0)')
    parser.add_argument('--attention_dropout', type=float, default=argparse.SUPPRESS,
                        help='Attention dropout rate (common: 0.0)')
    parser.add_argument('--embedding_dropout', type=float, default=argparse.SUPPRESS,
                        help='Embedding dropout rate (common: 0.0)')
    parser.add_argument('--learning_rate', type=float, default=argparse.SUPPRESS,
                        help='Learning rate (common: 0.001)')
    parser.add_argument('--embedding_dim', type=int, default=argparse.SUPPRESS,
                        help='Embedding dimension (common: 128)')
    parser.add_argument('--early_stop_patience', type=int, default=argparse.SUPPRESS,
                        help='Patience for early stopping (common: 2)')
    parser.add_argument('--reduce_lr_patience', type=int, default=argparse.SUPPRESS,
                        help='Patience for LR decay on plateau (common: 5)')
    parser.add_argument('--ffn_out_dim', type=int, default=argparse.SUPPRESS,
                        help='FFN hidden/output dimension in encoder layers (common: 512)')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        help='Random seed (common: 2026)')
    parser.add_argument('--embedding_regularizer', type=float, default=argparse.SUPPRESS,
                        help='Embedding regularizer L2 penalty (common: 1e-3)')
    parser.add_argument('--net_regularizer', type=float, default=argparse.SUPPRESS,
                        help='Net regularizer L2 penalty (common: 5e-5)')
    parser.add_argument('--metrics', nargs='+', default=argparse.SUPPRESS,
                        help='Metrics to evaluate (common: AUC logloss)')
    parser.add_argument('--use_cosine_lr', action='store_true', default=argparse.SUPPRESS,
                        help='Use CosineAnnealingLR scheduler with eta_min=lr/20')
    parser.add_argument('--double_tokens', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=argparse.SUPPRESS,
                        help='Double token count by split/reshape after embedding')
    args = parser.parse_args()

    if args.expid:
        experiment_id = args.expid
    else:
        if not args.dataset:
            parser.error("--dataset is required when --expid is not provided")
        class_name = MODEL_CLASS_MAP[MODEL_NAME]
        experiment_id = f"{class_name}_{args.dataset}_unified"

    # Build overrides dict: only include args that were explicitly provided
    cli_overrides = {}
    for key in [
        "epochs", "batch_size", "net_dropout", "transformer_dropout",
        "attention_dropout", "embedding_dropout", "learning_rate",
        "embedding_dim", "ffn_out_dim", "seed", "embedding_regularizer",
        "net_regularizer", "num_workers", "metrics", "use_cosine_lr",
        "double_tokens", "early_stop_patience", "reduce_lr_patience",
    ]:
        if hasattr(args, key):
            cli_overrides[key] = getattr(args, key)

    ret = run_experiment(
        MODEL_NAME, experiment_id, args.gpu,
        **cli_overrides
    )
    sys.exit(ret)


if __name__ == '__main__':
    main()
