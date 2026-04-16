#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch runner for all unified large-scale experiments.

Usage:
  # Run all models on all datasets
  python run_all_unified.py

  # Run specific model
  python run_all_unified.py --model HeteroAttention

  # Run specific dataset
  python run_all_unified.py --dataset frappe_x1

  # Run with GPU
  python run_all_unified.py --gpu 0

  # Dry run (print commands without executing)
  python run_all_unified.py --dry-run
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"

MODELS = ["HeteroAttention", "RankMixer", "HiFormer", "FAT", "TokenMixer_Large", "UniMixer_lite"]
DATASETS = ["movielenslatest_x1", "frappe_x1", "kuaivideo_x1", "taobaoad_x1"]

CLASS_NAMES = {
    "HeteroAttention": "HeteroAttention",
    "RankMixer": "RankMixer",
    "HiFormer": "HiFormer",
    "FAT": "FAT",
    "TokenMixer_Large": "TokenMixerLarge",
    "UniMixer_lite": "UniMixerLite",
}


def run_experiment(model_name, dataset_id, gpu, dry_run):
    class_name = CLASS_NAMES[model_name]
    exp_id = f"{class_name}_{dataset_id}_unified"
    model_dir = FUXICTR_ROOT / "model_zoo" / model_name
    run_script = model_dir / "run_expid.py"
    config_dir = model_dir / "config"

    cmd = [
        sys.executable,
        str(run_script),
        "--config", str(config_dir),
        "--expid", exp_id,
        "--gpu", str(gpu),
    ]

    print(f"\n{'='*70}")
    print(f"[{model_name}] -> {dataset_id} | expid={exp_id}")
    print(f"{'='*70}")

    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0

    result = subprocess.run(cmd, cwd=str(model_dir))
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset id or 'all'")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    target_models = MODELS if args.model == "all" else [args.model]
    target_datasets = DATASETS if args.dataset == "all" else [args.dataset]

    results = []
    for model_name in target_models:
        if model_name not in MODELS:
            print(f"[ERROR] Unknown model: {model_name}")
            continue
        for dataset_id in target_datasets:
            if dataset_id not in DATASETS:
                print(f"[ERROR] Unknown dataset: {dataset_id}")
                continue
            ret = run_experiment(model_name, dataset_id, args.gpu, args.dry_run)
            status = "OK" if ret == 0 else f"FAILED (code={ret})"
            results.append((model_name, dataset_id, status))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model_name, dataset_id, status in results:
        print(f"  {model_name:20s} | {dataset_id:20s} | {status}")


if __name__ == "__main__":
    main()
