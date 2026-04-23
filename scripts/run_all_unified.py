#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch runner for all unified large-scale experiments.

Usage:
  # Run all models on all datasets
  python scripts/run_all_unified.py

  # Run specific model
  python scripts/run_all_unified.py --model HeteroAttention

  # Run specific dataset
  python scripts/run_all_unified.py --dataset frappe_x1

  # Run with GPU
  python scripts/run_all_unified.py --gpu 0

  # Dry run (print commands without executing)
  python scripts/run_all_unified.py --dry-run
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "main"))

from run_expid import run_experiment, MODEL_CLASS_MAP

MODELS = ["HeteroAttention", "RankMixer", "HiFormer", "FAT", "TokenMixer_Large", "UniMixer_lite", "TransformerCTR"]
DATASETS = ["movielenslatest_x1", "frappe_x1", "kuaivideo_x1", "taobaoad_x1", "microvideo1.7m_x1"]


def run_single(model_name, dataset_id, gpu, dry_run, use_model_entry=False):
    class_name = MODEL_CLASS_MAP[model_name]
    exp_id = f"{class_name}_{dataset_id}_unified"

    if use_model_entry:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "FuxiCTR" / "model_zoo" / model_name / "run_expid.py"),
            "--expid", exp_id,
            "--gpu", str(gpu),
        ]
    else:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "main" / "run_expid.py"),
            "--model", model_name,
            "--expid", exp_id,
            "--gpu", str(gpu),
        ]

    print(f"\n{'='*70}")
    entry = "model-entry" if use_model_entry else "main-entry"
    print(f"[{model_name}] -> {dataset_id} | expid={exp_id} | via={entry}")
    print(f"{'='*70}")

    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0

    if use_model_entry:
        # Use subprocess for model entry to avoid sys.path/src pollution
        env = os.environ.copy()
        if gpu >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT / "FuxiCTR" / "model_zoo" / model_name), env=env)
        return result.returncode
    else:
        # Direct function call via main entry
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            inner_gpu = 0
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            inner_gpu = -1
        return run_experiment(model_name, exp_id, inner_gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset id or 'all'")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--use-model-entry", action="store_true",
                        help="Run via each model's own run_expid.py instead of main/run_expid.py")
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
            ret = run_single(model_name, dataset_id, args.gpu, args.dry_run, args.use_model_entry)
            status = "OK" if ret == 0 else f"FAILED (code={ret})"
            results.append((model_name, dataset_id, status))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model_name, dataset_id, status in results:
        print(f"  {model_name:20s} | {dataset_id:20s} | {status}")


if __name__ == "__main__":
    main()
