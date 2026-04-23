#!/usr/bin/env python3
"""
Hyperparameter search for FuxiCTR models with multi-GPU support.
Based on main/run_expid.py.

Usage:
    # Grid search on 4 GPUs for all models and datasets
    python main/hyperparam_search.py \
        --models RankMixer FAT HiFormer HeteroAttention TransformerCTR \
        --datasets frappe_x1 taobaoad_x1 \
        --gpus 0 1 2 3

    # Random search with 30 trials per (model, dataset)
    python main/hyperparam_search.py \
        --models RankMixer FAT \
        --datasets taobaoad_x1 \
        --search_mode random --n_trials 30 \
        --gpus 0 1 2 3

    # Override search space from CLI
    python main/hyperparam_search.py \
        --models RankMixer \
        --datasets frappe_x1 \
        --embedding_dims 32 64 128 \
        --ffn_out_dims 256 512 \
        --net_regularizers 0 1e-5 \
        --embedding_regularizers 0 1e-3 \
        --gpus 0
"""

import os
import sys
import json
import argparse
import itertools
import random
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"
RESULTS_DIR = PROJECT_ROOT / "results"
SEARCH_RESULTS_DIR = PROJECT_ROOT / "hyperparam_search_results"

sys.path.insert(0, str(FUXICTR_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from run_expid import run_experiment, MODEL_CLASS_MAP


# ---------------------------------------------------------------------------
# Default search space (user-defined)
# ---------------------------------------------------------------------------
DEFAULT_SEARCH_SPACE = {
    "frappe_x1": {
        "embedding_dim": [32, 64, 128],
        "ffn_out_dim": [256, 512],
        "net_regularizer": [5e-5, 5e-4, 5e-3],
        "embedding_regularizer": [5e-4, 1e-3, 5e-3, 1e-2],
    },
    "taobaoad_x1": {
        "embedding_dim": [32, 64, 128],
        "ffn_out_dim": [256, 512],
        "net_regularizer": [5e-5, 5e-4, 5e-3],
        "embedding_regularizer": [5e-4, 1e-3, 5e-3, 1e-2],
    },
}


def parse_result_line(line):
    """Parse a single result line from results CSV."""
    parts = line.strip().split(",[")
    if len(parts) < 6:
        return None

    def extract(tag, text):
        if text.startswith(f"{tag}] "):
            return text[len(f"{tag}] "):]
        return text

    timestamp = parts[0].strip()
    command = extract("command", parts[1])
    exp_id = extract("exp_id", parts[2])
    dataset_id = extract("dataset_id", parts[3])
    val_str = extract("val", parts[5])
    test_str = extract("test", parts[6]) if len(parts) > 6 else ""

    def parse_metrics(metric_str):
        metrics = {}
        for item in metric_str.split(" - "):
            item = item.strip()
            if ":" in item:
                k, v = item.split(":", 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        return metrics

    return {
        "timestamp": timestamp,
        "command": command,
        "exp_id": exp_id,
        "dataset_id": dataset_id,
        "val_metrics": parse_metrics(val_str),
        "test_metrics": parse_metrics(test_str),
    }


def read_latest_result(model_name, dataset_id):
    result_file = RESULTS_DIR / f"{model_name}_{dataset_id}_results.csv"
    if not result_file.exists():
        return None
    with open(result_file, "r") as f:
        lines = f.readlines()
    if not lines:
        return None
    return parse_result_line(lines[-1])


def get_best_metric(val_metrics):
    """Always use AUC as the primary metric (higher is better)."""
    if "AUC" in val_metrics:
        return val_metrics["AUC"]
    if val_metrics:
        return list(val_metrics.values())[0]
    return None


def generate_grid(search_space):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def generate_random(search_space, n_trials, seed=42):
    rng = random.Random(seed)
    all_combos = list(generate_grid(search_space))
    if n_trials >= len(all_combos):
        return all_combos
    return rng.sample(all_combos, n_trials)


def run_single_trial(model_name, dataset_id, hyperparams, gpu, seed=2024):
    class_name = MODEL_CLASS_MAP[model_name]
    exp_id = f"{class_name}_{dataset_id}_unified"

    hp = dict(hyperparams)
    # Workaround: model configs use 'ffn_dim' but CLI uses 'ffn_out_dim'.
    # The model source pops 'ffn_dim' from kwargs and overrides ffn_out_dim,
    # so we must ensure both keys carry the same value.
    if "ffn_out_dim" in hp:
        hp["ffn_dim"] = hp["ffn_out_dim"]
    hp["seed"] = seed

    print(f"\n[GPU:{gpu}] Model: {model_name} | Dataset: {dataset_id} | HP: {json.dumps(hp, ensure_ascii=False)}")

    ret = run_experiment(
        model_name=model_name,
        experiment_id=exp_id,
        gpu=gpu,
        **hp,
    )

    if ret != 0:
        print(f"[WARNING] Experiment returned non-zero code: {ret}")

    result = read_latest_result(model_name, dataset_id)
    return result


def search_one_model_dataset(model_name, dataset_id, search_space, search_mode,
                             n_trials, gpu, seed, output_queue):
    """Run all trials for ONE (model, dataset) on ONE GPU. Called in a subprocess."""
    if search_mode == "grid":
        trials = list(generate_grid(search_space))
    elif search_mode == "random":
        trials = generate_random(search_space, n_trials, seed)
    else:
        raise ValueError(f"Unknown search_mode: {search_mode}")

    local_best = None
    local_best_score = -float("inf")
    all_results = []

    for trial_idx, hyperparams in enumerate(trials, 1):
        result = run_single_trial(model_name, dataset_id, hyperparams, gpu=gpu, seed=seed + trial_idx)
        if result is None:
            print(f"[ERROR] No result for {model_name}/{dataset_id} trial {trial_idx}")
            continue

        val_metrics = result.get("val_metrics", {})
        score = get_best_metric(val_metrics)
        logloss = val_metrics.get("logloss", None)

        record = {
            "model": model_name,
            "dataset": dataset_id,
            "trial": trial_idx,
            "hyperparams": hyperparams,
            "val_metrics": val_metrics,
            "test_metrics": result.get("test_metrics", {}),
            "score": score,
            "timestamp": result.get("timestamp"),
        }
        all_results.append(record)

        if score is not None and score > local_best_score:
            local_best_score = score
            local_best = record
            print(f"[GPU:{gpu}] *** New best {model_name}/{dataset_id}: AUC={score:.6f}, logloss={logloss}")

    output_queue.put({
        "model": model_name,
        "dataset": dataset_id,
        "best": local_best,
        "all_results": all_results,
    })


def gpu_worker(task_queue, gpu_id, seed, output_queue):
    """Worker process bound to one GPU. Pulls tasks until queue is empty."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker GPU {gpu_id}] Started")

    while True:
        try:
            task = task_queue.get(timeout=5)
        except Exception:
            break

        if task is None:
            break

        model_name, dataset_id, space, search_mode, n_trials = task
        print(f"[Worker GPU {gpu_id}] Task: {model_name}/{dataset_id} | Mode: {search_mode}")
        try:
            search_one_model_dataset(
                model_name, dataset_id, space, search_mode,
                n_trials, gpu=0, seed=seed, output_queue=output_queue
            )
        except Exception as e:
            print(f"[Worker GPU {gpu_id}] Error on {model_name}/{dataset_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"[Worker GPU {gpu_id}] Finished")


def search(model_list, dataset_list, search_space_map, search_mode, n_trials, gpus, seed, output_dir=None):
    if output_dir is None:
        output_dir = SEARCH_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list: one task = all trials for one (model, dataset)
    tasks = []
    for model_name in model_list:
        for dataset_id in dataset_list:
            space = search_space_map.get(dataset_id, {})
            if not space:
                print(f"[SKIP] No search space for {dataset_id}")
                continue
            tasks.append((model_name, dataset_id, space, search_mode, n_trials))

    print(f"Total tasks (model x dataset): {len(tasks)}")
    print(f"GPUs available: {gpus}")
    print(f"Search mode: {search_mode}")
    if search_mode == "grid" and tasks:
        per_task = len(list(generate_grid(tasks[0][2])))
        print(f"Trials per task: {per_task}")
        print(f"Total trials: {len(tasks) * per_task}")
    elif search_mode == "random":
        print(f"Trials per task: {n_trials}")
        print(f"Total trials: {len(tasks) * n_trials}")

    manager = mp.Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()

    for task in tasks:
        task_queue.put(task)

    # Start one worker per GPU
    processes = []
    for gpu_id in gpus:
        p = mp.Process(target=gpu_worker, args=(task_queue, gpu_id, seed, results_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect results
    all_results = []
    best_configs = {}
    while not results_queue.empty():
        res = results_queue.get()
        all_results.extend(res["all_results"])
        best_configs[(res["model"], res["dataset"])] = res["best"]

    # Save summary
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = output_dir / f"search_summary_{timestamp_str}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "all_results": all_results,
            "best_configs": {
                f"{k[0]}_{k[1]}": v for k, v in best_configs.items()
            },
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[INFO] Search summary saved to {summary_path}")

    csv_path = output_dir / f"search_summary_{timestamp_str}.csv"
    with open(csv_path, "w") as f:
        f.write("model,dataset,trial,embedding_dim,ffn_out_dim,net_regularizer,embedding_regularizer,val_AUC,val_logloss\n")
        for r in all_results:
            hp = r["hyperparams"]
            f.write(f"{r['model']},{r['dataset']},{r['trial']},"
                    f"{hp.get('embedding_dim','')},{hp.get('ffn_out_dim','')},"
                    f"{hp.get('net_regularizer','')},{hp.get('embedding_regularizer','')},"
                    f"{r['score']},{r['val_metrics'].get('logloss','')}\n")
    print(f"[INFO] CSV summary saved to {csv_path}")

    # Print best configs
    print("\n" + "="*70)
    print(" BEST CONFIGURATIONS SUMMARY (by AUC) ")
    print("="*70)
    for (model_name, dataset_id), best in sorted(best_configs.items()):
        if best is None:
            print(f"\n{model_name} / {dataset_id}: NO VALID RESULT")
            continue
        print(f"\n>> {model_name} / {dataset_id}")
        print(f"   Best AUC (val): {best['score']:.6f}")
        print(f"   Val metrics: {best['val_metrics']}")
        print(f"   Hyperparams:")
        for k, v in best["hyperparams"].items():
            print(f"      {k}: {v}")

    return best_configs


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for FuxiCTR models (multi-GPU)")
    parser.add_argument("--models", nargs="+", required=True,
                        choices=list(MODEL_CLASS_MAP.keys()),
                        help="Model names to search")
    parser.add_argument("--datasets", nargs="+", required=True,
                        choices=["frappe_x1", "movielenslatest_x1", "kuaivideo_x1",
                                 "taobaoad_x1", "microvideo1.7m_x1"],
                        help="Dataset IDs to search")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0],
                        help="List of GPU indices to use, e.g. 0 1 2 3")
    parser.add_argument("--search_mode", type=str, default="grid",
                        choices=["grid", "random"],
                        help="Search strategy (default: grid)")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of random trials (only for random mode)")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed for reproducibility")

    # Allow explicit override of search space from CLI
    parser.add_argument("--embedding_dims", nargs="+", type=int, default=None)
    parser.add_argument("--ffn_out_dims", nargs="+", type=int, default=None)
    parser.add_argument("--net_regularizers", nargs="+", type=float, default=None)
    parser.add_argument("--embedding_regularizers", nargs="+", type=float, default=None)

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save search results")

    args = parser.parse_args()

    # Build search space
    search_space_map = {}
    for ds in args.datasets:
        space = dict(DEFAULT_SEARCH_SPACE.get(ds, {}))
        if args.embedding_dims is not None:
            space["embedding_dim"] = args.embedding_dims
        if args.ffn_out_dims is not None:
            space["ffn_out_dim"] = args.ffn_out_dims
        if args.net_regularizers is not None:
            space["net_regularizer"] = args.net_regularizers
        if args.embedding_regularizers is not None:
            space["embedding_regularizer"] = args.embedding_regularizers
        search_space_map[ds] = space

    search(
        model_list=args.models,
        dataset_list=args.datasets,
        search_space_map=search_space_map,
        search_mode=args.search_mode,
        n_trials=args.n_trials,
        gpus=args.gpus,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
