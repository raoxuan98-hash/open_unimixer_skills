#!/usr/bin/env python3
"""
Preprocess dataset for FuxiCTR training.

This script builds feature_map.json, feature_processor.pkl, vocab files,
and parquet caches ahead of time, so that model training can start immediately.

Usage:
    python prepare_train_data.py --dataset kuaivideo_x1
    python prepare_train_data.py --dataset kuaivideo_x1 --force
    python prepare_train_data.py --dataset kuaivideo_x1 --data_block_size 500000
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"

sys.path.insert(0, str(FUXICTR_ROOT))
from fuxictr.preprocess import FeatureProcessor, build_dataset

DATASETS = {
    "frappe_x1": "Frappe",
    "movielenslatest_x1": "MovieLens Latest",
    "kuaivideo_x1": "KuaiVideo",
    "taobaoad_x1": "TaobaoAd",
    "microvideo1.7m_x1": "MicroVideo1.7M",
}


def prepare_dataset(dataset_id, force=False, data_block_size=1000000):
    if dataset_id not in DATASETS:
        print(f"[ERROR] Unknown dataset: {dataset_id}")
        print(f"Supported: {', '.join(DATASETS.keys())}")
        return 1

    if not DATASET_CONFIG_PATH.exists():
        print(f"[ERROR] Unified dataset config not found: {DATASET_CONFIG_PATH}")
        return 1

    with open(DATASET_CONFIG_PATH, "r") as f:
        dataset_configs = yaml.safe_load(f)

    if dataset_id not in dataset_configs:
        print(f"[ERROR] Dataset {dataset_id} not found in {DATASET_CONFIG_PATH}")
        return 1

    params = dataset_configs[dataset_id].copy()
    params["dataset_id"] = dataset_id
    params.setdefault("data_block_size", data_block_size)
    params.setdefault("rebuild_dataset", True)

    data_dir = os.path.join(params["data_root"], dataset_id)
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    if os.path.exists(feature_map_json) and not force:
        print(f"[INFO] {dataset_id} already prepared at {data_dir}. Use --force to rebuild.")
        return 0

    if force and os.path.exists(data_dir):
        print(f"[INFO] Force rebuild: removing old cache in {data_dir}")
        for f in Path(data_dir).glob("*"):
            if f.is_file():
                f.unlink()

    print(f"[INFO] Preparing {DATASETS[dataset_id]} ({dataset_id}) ...")
    print(f"[INFO] data_block_size={params.get('data_block_size', 0)}")

    feature_encoder = FeatureProcessor(**params)
    build_dataset(feature_encoder, **params)

    print(f"[DONE] {dataset_id} prepared successfully at {data_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Preprocess FuxiCTR datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID, e.g. kuaivideo_x1")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cache exists")
    parser.add_argument("--data_block_size", type=int, default=1000000,
                        help="Block size for parquet conversion")
    args = parser.parse_args()

    ret = prepare_dataset(args.dataset, args.force, args.data_block_size)
    sys.exit(ret)


if __name__ == "__main__":
    main()
