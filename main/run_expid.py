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
Unified training entry for all FuxiCTR models.
Replaces model-specific run_expid.py and dataset-specific train_xxx.py.

Usage:
    python main/run_expid.py --model RankMixer --dataset frappe_x1 --gpu 0
    python main/run_expid.py --model RankMixer --expid RankMixer_frappe_x1_unified --gpu 0
"""
import numpy as np
import os
import sys
import logging
import gc
import argparse
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"
UNIFIED_DATASET_CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(FUXICTR_ROOT))

from fuxictr.utils import load_model_config, load_dataset_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything


MODEL_CLASS_MAP = {
    "DCNv2": "DCNv2",
    "Wukong": "WuKong",
    "FAT": "FAT",
    "HiFormer": "HiFormer",
    "TransformerCTR": "TransformerCTR",
    "HeteroAttention": "HeteroAttention",
    "RankMixer": "RankMixer",
    "RankMixer_Norm": "RankMixer",
    "TokenMixer_Large": "TokenMixerLarge",
    "UniMixer_norm": "UniMixerNorm",
    "UniMixer_norm_linear": "UniMixerNorm",
    "UniMixer_layernorm": "UniMixerLayerNorm",
    "UniMixer_lite": "UniMixerLite",
    "UniMixer_lite_sinkhorn": "UniMixerLite",
    "UniMixer_norm_lite_fusion": "UniMixerLite",
    "UniMixer_norm_lite_fusion_xonly": "UniMixerLite",
    "HybridMixer": "HybridMixer",
}

# Fair alignment config: same hyperparameters per dataset across all models.
# When bars_mode=False (default), these values override model-specific configs.
UNIFIED_ALIGNMENT = {
    "frappe_x1": {
        "epochs": 10,
        "batch_size": 1024,
        "embedding_dim": 60,
        "ffn_in_dim": None,
        "ffn_out_dim": 256,
        "net_dropout": 0.1,
        "embedding_regularizer": 0.001,
        "net_regularizer": 0.0,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },

    "movielenslatest_x1": {
        "epochs": 20,
        "batch_size": 4096,
        "embedding_dim": 120,
        "ffn_in_dim": None,
        "ffn_out_dim": 256,
        "net_dropout": 0.1,
        "embedding_regularizer": 0.001,
        "net_regularizer": 0.0,
        # "net_regularizer": 1e-3,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },

    "kuaivideo_x1": {
        "epochs": 3,
        "batch_size": 8192,
        "embedding_dim": 64,
        "ffn_in_dim": 140,
        "ffn_out_dim": 256,
        "net_dropout": 0.1,
        "embedding_regularizer": 0.001,
        "net_regularizer": 0.0,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },

    "taobaoad_x1": {
        "epochs": 2,
        "batch_size": 8192,
        "embedding_dim": 140,
        "ffn_in_dim": None,
        "ffn_out_dim": 512,
        "net_dropout": 0.1,
        "embedding_regularizer": 0.001,
        "net_regularizer": 3e-5,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },

    "microvideo1.7m_x1": {
        "epochs": 3,
        "batch_size": 4096,
        "embedding_dim": 64,
        "ffn_in_dim": 60,
        "ffn_out_dim": 256,
        "net_dropout": 0.1,
        "embedding_regularizer": 0.001,
        "net_regularizer": 0.0,
        "use_pos_embedding": False,
        "use_cosine_lr": False,
    },
}


def import_model_src(model_name):
    """Dynamically import the src package from a specific model directory."""
    model_dir = FUXICTR_ROOT / "model_zoo" / model_name
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


def adjust_sinkhorn_temperature(model, model_name, dataset_id, epoch, epochs, batch_index=None, steps_per_epoch=None):
    """Adjust sinkhorn temperature for supported models based on dataset and epoch."""
    if model_name not in ("UniMixer_lite", "UniMixer_lite_sinkhorn", "UniMixer_norm_lite_fusion", "UniMixer_norm_lite_fusion_xonly", "HybridMixer"):
        return
    
    threshold_epoch = 4 if dataset_id in ("frappe_x1", "movielenslatest_x1") else 1
    
    if epoch < threshold_epoch:
        return
    
    if batch_index is None:
        # Beginning-of-epoch adjustment
        temp = 0.05
        model.set_sinkhorn_temperature(temp)
        logging.info(f"Epoch {epoch+1}/{epochs}: Setting sinkhorn_temperature to {temp}")

    elif steps_per_epoch is not None and batch_index == steps_per_epoch // 2:
        model.set_sinkhorn_temperature(0.05)
        logging.info(f"Epoch {epoch+1} step {batch_index+1}/{steps_per_epoch}: "
                     f"Setting sinkhorn_temperature to 0.05")


def run_experiment(model_name, experiment_id, gpu=-1, dataset_config_dir=None,
                    epochs=None, bars_mode=False,
                    batch_size=None, net_dropout=None, transformer_dropout=None,
                    attention_dropout=None, embedding_dropout=None,
                    learning_rate=None, embedding_dim=None, ffn_in_dim=None,
                    ffn_out_dim=None, seed=None,
                    embedding_regularizer=None, net_regularizer=None,
                    num_workers=None, metrics=None, use_cosine_lr=None,
                    double_tokens=None, use_basis=None, basis_num=None,
                    block_size=None, seq_ffn_basis_num=None, token_num_after_reshape=None,
                    early_stop_patience=None,
                    reduce_lr_patience=None,
                    reduce_lr_factor=None, optimizer=None):
                    
    if model_name not in MODEL_CLASS_MAP:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CLASS_MAP.keys())}")
    
    model_dir = FUXICTR_ROOT / "model_zoo" / model_name
    model_config_dir = model_dir / "config"
    
    # 1. Load model config
    model_params = load_model_config(str(model_config_dir), experiment_id)
    dataset_id = model_params["dataset_id"]
    
    # 1.5 Apply fair alignment overrides (unless bars_mode is enabled)
    if not bars_mode and dataset_id in UNIFIED_ALIGNMENT:
        for key, value in UNIFIED_ALIGNMENT[dataset_id].items():
            model_params[key] = value
        # Remove legacy ffn_dim to avoid shadowing ffn_out_dim
        if "ffn_out_dim" in UNIFIED_ALIGNMENT[dataset_id]:
            model_params.pop("ffn_dim", None)
    
    # 2. Load dataset config from unified source (independent of model)
    if dataset_config_dir is None:
        dataset_config_dir = str(UNIFIED_DATASET_CONFIG_DIR)
    else:
        dataset_config_dir = str(dataset_config_dir)
    if not Path(dataset_config_dir, "dataset_config.yaml").exists():
        raise FileNotFoundError(
            f"Dataset config not found at {Path(dataset_config_dir) / 'dataset_config.yaml'}. "
            f"Please ensure it exists."
        )
    data_params = load_dataset_config(dataset_config_dir, dataset_id)
    
    # Merge: model_params override data_params
    params = dict(data_params, **model_params)
    params["gpu"] = gpu
    params.setdefault("data_format", "csv")
    
    # ---- Apply command-line overrides (highest priority) ----
    cli_overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "net_dropout": net_dropout,
        "transformer_dropout": transformer_dropout,
        "attention_dropout": attention_dropout,
        "embedding_dropout": embedding_dropout,
        "learning_rate": learning_rate,
        "embedding_dim": embedding_dim,
        "ffn_in_dim": ffn_in_dim,
        "ffn_out_dim": ffn_out_dim,
        "seed": seed,
        "embedding_regularizer": embedding_regularizer,
        "net_regularizer": net_regularizer,
        "optimizer": optimizer,
        "num_workers": num_workers,
        "metrics": metrics,
        "use_cosine_lr": use_cosine_lr,
        "double_tokens": double_tokens,
        "use_basis": use_basis,
        "basis_num": basis_num,
        "block_size": block_size,
        "seq_ffn_basis_num": seq_ffn_basis_num,
        "token_num_after_reshape": token_num_after_reshape,
        "early_stop_patience": early_stop_patience,
        "reduce_lr_patience": reduce_lr_patience,
        "reduce_lr_factor": reduce_lr_factor,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            params[key] = value

    # Workaround: model configs use 'ffn_dim' but CLI uses 'ffn_out_dim'.
    # The model source pops 'ffn_dim' from kwargs and overrides ffn_out_dim,
    # so we must ensure CLI ffn_out_dim wins by removing ffn_dim from params.
    if params.get("ffn_out_dim") is not None:
        params.pop("ffn_dim", None)

    # 3. Dataset preparation check (strict separation)
    data_dir = os.path.join(params["data_root"], dataset_id)
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    
    if params["data_format"] == "csv":
        if not os.path.exists(feature_map_json):
            raise RuntimeError(
                f"Dataset {dataset_id} is not prepared. "
                f"Please run: python prepare_train_data.py --dataset {dataset_id}"
            )
        # Parquet paths are deterministic after preprocessing
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
    model.count_parameters()  # print number of parameters used in model
    
    # ---- Parameter & FLOPs statistics ----
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "embedding" in n
    )
    dense_params = total_params - embedding_params
    logging.info(f"Dense parameters (non-embedding): {dense_params:,}")
    
    # Per-sample FLOPs estimation using a dummy batch of size 1
    def _make_dummy_batch(feature_map, batch_size=1, device="cpu"):
        dummy = {}
        for name, spec in feature_map.features.items():
            feat_type = spec.get("type", "categorical")
            if feat_type == "meta":
                continue
            if feat_type == "numeric":
                dummy[name] = torch.randn(batch_size, 1, device=device)
            elif feat_type == "sequence":
                max_len = spec.get("max_len", 10)
                vocab_size = spec.get("vocab_size", 100)
                dummy[name] = torch.randint(
                    0, vocab_size, (batch_size, max_len), device=device
                )
            else:  # categorical
                vocab_size = spec.get("vocab_size", 100)
                dummy[name] = torch.randint(
                    0, vocab_size, (batch_size,), device=device
                )
        return dummy
    
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        was_training = model.training
        model.eval()
        dummy_batch = _make_dummy_batch(feature_map, batch_size=1, device=model.device)
        with FlopCounterMode(display=False) as flop_counter:
            _ = model(dummy_batch)
            flops = flop_counter.get_total_flops()
        logging.info(f"FLOPs per sample: {flops:,.0f}")
        if was_training:
            model.train()
    except Exception as e:
        logging.warning(f"FLOPs estimation skipped: {e}")
    
    # 7. Train (use streaming/block loader for large datasets to avoid OOM/hanging)
    params["streaming"] = True
    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    
    epochs = params.get("epochs", 100)
    steps_per_epoch = len(train_gen)
    
    # Initialize training states
    model._max_gradient_norm = params.get("max_gradient_norm", 10.)
    model._steps_per_epoch = steps_per_epoch
    model._total_steps = 0
    model._batch_index = 0
    model._epoch_index = 0
    
    best_metric = -np.inf if model._monitor_mode == "max" else np.inf
    best_epoch = -1
    model._stopping_steps = 0
    early_stop_patience = getattr(model, '_early_stop_patience', None)
    min_delta = 1e-6
    
    logging.info("Start training: {} batches/epoch, {} epochs".format(steps_per_epoch, epochs))
    for epoch in range(epochs):
        model._epoch_index = epoch
        model.train()
        train_loss = 0
        ema_train_loss = 0
        ema_decay = 0.96
        
        # Temperature adjustment logic for UniMixerLite/HybridMixer
        adjust_sinkhorn_temperature(model, model_name, dataset_id, epoch, epochs)
        
        logging.info("************ Epoch={} start ************".format(epoch + 1))
        epoch_iterator = tqdm(train_gen, total=steps_per_epoch,
                              desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for batch_index, batch_data in enumerate(epoch_iterator):
            model._batch_index = batch_index
            model._total_steps += 1
            
            # Temperature adjustment: drop temp to 0.05 at halfway of first epoch
            adjust_sinkhorn_temperature(model, model_name, dataset_id, epoch, epochs,
                                        batch_index=batch_index, steps_per_epoch=steps_per_epoch)
            
            loss = model.train_step(batch_data)
            loss_item = loss.item()
            train_loss += loss_item
            
            if batch_index == 0:
                ema_train_loss = loss_item
            else:
                ema_train_loss = ema_decay * ema_train_loss + (1 - ema_decay) * loss_item
            
            epoch_iterator.set_postfix(ema_loss=f"{ema_train_loss:.6f}")
        
        logging.info("************ Epoch={} end, avg_loss={:.6f} ************".format(
            epoch + 1, train_loss / steps_per_epoch))
        
        # Evaluate on validation set after each epoch and save best checkpoint
        logging.info('****** Validation evaluation @epoch {} ******'.format(epoch + 1))
        valid_result_epoch = model.evaluate(valid_gen)
        monitor_value = model._monitor.get_value(valid_result_epoch)
        improved = (model._monitor_mode == "max" and monitor_value > best_metric + min_delta) or \
                   (model._monitor_mode == "min" and monitor_value < best_metric - min_delta)
        if improved:
            best_metric = monitor_value
            best_epoch = epoch
            model._stopping_steps = 0
            os.makedirs(os.path.dirname(model.checkpoint), exist_ok=True)
            model.save_weights(model.checkpoint)
            logging.info("Save best model @epoch {}: monitor({})={:.6f}".format(
                epoch + 1, model._monitor_mode, monitor_value))
        else:
            model._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP! stopping_steps={}".format(
                model._monitor_mode, monitor_value, model._stopping_steps))
            if early_stop_patience is not None and model._stopping_steps >= early_stop_patience:
                logging.info("********* Epoch={} early stop *********".format(epoch + 1))
                break
    
    logging.info("Training finished.")
    
    # Load best checkpoint after all epochs
    if os.path.exists(model.checkpoint):
        logging.info("Load best model from epoch {}: {}".format(best_epoch + 1, model.checkpoint))
        model.load_weights(model.checkpoint)
    else:
        logging.warning("No best model checkpoint found.")
    
    logging.info('****** Validation evaluation (best model) ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    test_result = {}
    if params.get("test_data"):
        logging.info('******** Test evaluation (best model) ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)
    
    # 8. Save results
    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    result_filename = result_dir / f"{model_name}_{dataset_id}_results.csv"
    flops_str = f"{flops:.0f}" if flops is not None else "N.A."
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {},[dense_params] {},[flops] {}\n'
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, dataset_id,
                    "N.A.", print_to_list(valid_result), print_to_list(test_result),
                    dense_params, flops_str))
    logging.info(f"Results saved to {result_filename}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Unified FuxiCTR training entry")
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g. RankMixer')
    parser.add_argument('--expid', type=str, default=None, help='Experiment id to run')
    AVAILABLE_DATASETS = ["frappe_x1", "movielenslatest_x1", "kuaivideo_x1", "taobaoad_x1", "microvideo1.7m_x1"]
    parser.add_argument('--dataset', type=str, default=None, choices=AVAILABLE_DATASETS, help=f"Dataset id. Choices: {', '.join(AVAILABLE_DATASETS)}")
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu (default: 0)')
    parser.add_argument('--dataset_config', type=str, default=None, help='Override dataset config directory (default: configs/)')
    parser.add_argument('--bars_mode', action='store_true', help='Use BARS original config without unified overrides')

    # Training hyperparameters: use argparse.SUPPRESS so missing CLI args do NOT
    # override the values in model_config.yaml.
    parser.add_argument('--num_workers', type=int, default=argparse.SUPPRESS,
                        help='Number of workers for data loading (common: 8)')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS,
                        help='Number of training epochs (common: 16)')
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        help='Batch size (common: 2048)')
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
    parser.add_argument('--ffn_in_dim', type=int, default=argparse.SUPPRESS,
                        help='FFN input dimension, projected from embedding_dim via nn.Linear (common: 128)')
    parser.add_argument('--early_stop_patience', type=int, default=argparse.SUPPRESS,
                        help='Patience for early stopping (common: 2)')
    parser.add_argument('--reduce_lr_patience', type=int, default=argparse.SUPPRESS,
                        help='Patience for LR decay on plateau (common: 5)')
    parser.add_argument('--reduce_lr_factor', type=float, default=argparse.SUPPRESS,
                        help='LR decay factor when plateau is reached (common: 0.5)')
    parser.add_argument('--ffn_out_dim', type=int, default=argparse.SUPPRESS,
                        help='FFN hidden/output dimension in encoder layers (common: 512)')
    parser.add_argument('--seed', type=int, default=argparse.SUPPRESS,
                        help='Random seed (common: 2026)')
    parser.add_argument('--embedding_regularizer', type=float, default=argparse.SUPPRESS,
                        help='Embedding regularizer L2 penalty (common: 1e-3)')
    parser.add_argument('--net_regularizer', type=float, default=argparse.SUPPRESS,
                        help='Net regularizer L2 penalty (common: 5e-5)')
    parser.add_argument('--optimizer', type=str, default=argparse.SUPPRESS,
                        choices=['adam', 'adamw'],
                        help='Optimizer to use: adam (explicit regularization in loss) or adamw (weight_decay)')
    parser.add_argument('--metrics', nargs='+', default=argparse.SUPPRESS,
                        help='Metrics to evaluate (common: AUC logloss)')
    parser.add_argument('--use_cosine_lr', action='store_true', default=argparse.SUPPRESS,
                        help='Use CosineAnnealingLR scheduler with eta_min=lr/20')
    parser.add_argument('--double_tokens', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=argparse.SUPPRESS,
                        help='Double token count by split/reshape after embedding')
    parser.add_argument('--use_basis', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=argparse.SUPPRESS,
                        help='Use basis-weighted spherical SwishGLU for parameter efficiency')
    parser.add_argument('--basis_num', type=int, default=argparse.SUPPRESS,
                        help='Number of basis matrices when use_basis=True (common: 8)')
    parser.add_argument('--block_size', type=int, default=argparse.SUPPRESS,
                        help='Chunk size for token re-partitioning in UniMixerBlock (common: 32)')
    parser.add_argument('--seq_ffn_basis_num', type=int, default=argparse.SUPPRESS,
                        help='Number of basis matrices for local interaction W (common: 4)')
    parser.add_argument('--token_num_after_reshape', type=int, default=argparse.SUPPRESS,
                        help='Hidden dimension of the global FFN inside UniMixerBlock (common: total_dim//block_size)')
    args = parser.parse_args()
    
    if args.expid:
        experiment_id = args.expid
    else:
        if not args.dataset:
            parser.error("--dataset is required when --expid is not provided")
        class_name = MODEL_CLASS_MAP.get(args.model)
        if not class_name:
            parser.error(f"Unknown model: {args.model}")
        experiment_id = f"{class_name}_{args.dataset}_unified"
    
    # Build overrides dict: only include args that were explicitly provided
    cli_overrides = {"dataset_config_dir": args.dataset_config, "bars_mode": args.bars_mode}
    for key in [
        "epochs", "batch_size", "net_dropout", "transformer_dropout",
        "attention_dropout", "embedding_dropout", "learning_rate",
        "embedding_dim", "ffn_in_dim", "ffn_out_dim", "seed", "embedding_regularizer",
        "net_regularizer", "num_workers", "metrics", "use_cosine_lr",
        "double_tokens", "use_basis", "basis_num",
        "block_size", "seq_ffn_basis_num", "token_num_after_reshape",
        "early_stop_patience", "reduce_lr_patience",
        "reduce_lr_factor", "optimizer",
    ]:
        if hasattr(args, key):
            cli_overrides[key] = getattr(args, key)

    ret = run_experiment(args.model, experiment_id, args.gpu, **cli_overrides)
    sys.exit(ret)


if __name__ == '__main__':
    main()
