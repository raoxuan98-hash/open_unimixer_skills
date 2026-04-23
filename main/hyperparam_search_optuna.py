#!/usr/bin/env python3
"""
Optuna-based hyperparameter search for FuxiCTR models (multi-GPU).
Each (model, dataset) combination gets its OWN Optuna study.
Model is NOT a hyperparameter; we search the best params for each model independently.

Usage:
    # 4 GPUs, 30 trials per (model, dataset)
    python main/hyperparam_search_optuna.py \
        --models RankMixer FAT HiFormer HeteroAttention TransformerCTR \
        --datasets frappe_x1 taobaoad_x1 movielenslatest_x1\
        --gpus 0 1 2 3 \
        --n_trials 30

    # Override search space
    python main/hyperparam_search_optuna.py \
        --models RankMixer \
        --datasets frappe_x1 \
        --embedding_dims 32 64 128 \
        --ffn_out_dims 256 512 \
        --net_regularizers 5e-5 5e-4 5e-3 \
        --embedding_regularizers 5e-4 1e-3 5e-3 1e-2 \
        --gpus 0 \
        --n_trials 20
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

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
        "embedding_dim": [16, 48, 144],
        "ffn_out_dim": [64, 128, 256, 512],
        "net_dropout": [0.0, 0.2],
        "net_regularizer": [5e-5, 5e-4, 5e-3],
        "embedding_regularizer": [5e-4, 1e-3, 5e-3, 1e-2],
    },
    "taobaoad_x1": {
        "embedding_dim": [48, 144],
        "ffn_out_dim": [128, 256, 512],
        "net_dropout": [0.0, 0.2],
        "net_regularizer": [5e-5, 5e-4, 5e-3],
        "embedding_regularizer": [5e-4, 1e-3, 5e-3, 1e-2],
    },
    "movielenslatest_x1": {
        "embedding_dim": [16, 48, 144],
        "ffn_out_dim": [64, 128, 256, 512],
        "net_dropout": [0.0, 0.2],
        "net_regularizer": [5e-5, 5e-4, 5e-3],
        "embedding_regularizer": [5e-4, 1e-3, 5e-3, 1e-2],
    },
    "microvideo1.7m_x1": {
        "embedding_dim": [64],  # fixed to 64 due to pretrained item embeddings
        "ffn_out_dim": [64, 128, 256, 512],
        "net_dropout": [0.0, 0.2],
        "net_regularizer": [5e-5, 5e-4, 5e-3],
        "embedding_regularizer": [5e-4, 1e-3, 5e-3, 1e-2],
    },
    "kuaivideo_x1": {
        "embedding_dim": [64],  # fixed to 64 due to pretrained item embeddings
        "ffn_out_dim": [64, 128, 256, 512],
        "net_dropout": [0.0, 0.2],
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


def run_single_trial(model_name, dataset_id, hyperparams, gpu, seed=2024, gpu_label=None):
    """Run one training trial. gpu_label is for logging only."""
    class_name = MODEL_CLASS_MAP[model_name]
    exp_id = f"{class_name}_{dataset_id}_unified"

    hp = dict(hyperparams)
    hp["seed"] = seed
    hp.setdefault("batch_size", 4096)

    label = gpu_label if gpu_label is not None else gpu
    print(f"\n[GPU:{label}] {model_name}/{dataset_id} | HP: {json.dumps(hp, ensure_ascii=False)}")

    ret = run_experiment(
        model_name=model_name,
        experiment_id=exp_id,
        gpu=gpu,
        **hp,
    )

    if ret != 0:
        print(f"[WARNING] Non-zero return: {ret}")

    result = read_latest_result(model_name, dataset_id)
    return result


def run_optuna_study(model_name, dataset_id, search_space, n_trials, gpu_id, seed, output_dir, overwrite=False):
    """Run one Optuna study for a single (model, dataset) on a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Each (model, dataset) gets its own journal file
    study_dir = Path(output_dir) / f"{model_name}_{dataset_id}"
    study_dir.mkdir(parents=True, exist_ok=True)
    journal_path = study_dir / "optuna_journal.log"

    if overwrite and journal_path.exists():
        journal_path.unlink()
        print(f"[GPU:{gpu_id}] Overwritten old journal: {journal_path}")

    storage = JournalStorage(JournalFileStorage(str(journal_path)))

    study_name = f"{model_name}_{dataset_id}_auc"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial):
        hp = {
            "embedding_dim": trial.suggest_categorical(
                "embedding_dim", search_space["embedding_dim"]
            ),
            "ffn_out_dim": trial.suggest_categorical(
                "ffn_out_dim", search_space["ffn_out_dim"]
            ),
            "net_dropout": trial.suggest_categorical(
                "net_dropout", search_space["net_dropout"]
            ),
            "net_regularizer": trial.suggest_categorical(
                "net_regularizer", search_space["net_regularizer"]
            ),
            "embedding_regularizer": trial.suggest_categorical(
                "embedding_regularizer", search_space["embedding_regularizer"]
            ),
        }
        result = run_single_trial(
            model_name, dataset_id, hp,
            gpu=0, seed=seed + trial.number, gpu_label=gpu_id
        )
        if result is None:
            raise optuna.TrialPruned()

        score = get_best_metric(result.get("val_metrics", {}))
        if score is None:
            raise optuna.TrialPruned()

        trial.set_user_attr("val_metrics", result.get("val_metrics", {}))
        trial.set_user_attr("test_metrics", result.get("test_metrics", {}))
        return score

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    return {
        "model": model_name,
        "dataset": dataset_id,
        "best_params": best.params,
        "best_value": best.value,
        "best_trial_number": best.number,
        "user_attrs": dict(best.user_attrs),
    }


def gpu_worker(task_queue, gpu_id, seed, output_dir, overwrite, results_queue):
    """Worker process bound to one GPU. Pulls studies until queue is empty."""
    print(f"[Worker GPU {gpu_id}] Started")

    while True:
        try:
            task = task_queue.get(timeout=5)
        except Exception:
            break
        if task is None:
            break

        model_name, dataset_id, space, n_trials = task
        print(f"[Worker GPU {gpu_id}] Study: {model_name}/{dataset_id} | Trials: {n_trials}")
        try:
            result = run_optuna_study(
                model_name, dataset_id, space, n_trials,
                gpu_id=gpu_id, seed=seed, output_dir=output_dir,
                overwrite=overwrite,
            )
            results_queue.put(result)
        except Exception as e:
            print(f"[Worker GPU {gpu_id}] Error on {model_name}/{dataset_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"[Worker GPU {gpu_id}] Finished")


def search(model_list, dataset_list, search_space_map, n_trials, gpus, seed, output_dir=None, overwrite=False):
    if output_dir is None:
        output_dir = SEARCH_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for model_name in model_list:
        for dataset_id in dataset_list:
            space = search_space_map.get(dataset_id, {})
            if not space:
                print(f"[SKIP] No search space for {dataset_id}")
                continue
            tasks.append((model_name, dataset_id, space, n_trials))

    print(f"Total studies (model x dataset): {len(tasks)}")
    print(f"Trials per study: {n_trials}")
    print(f"Total trials: {len(tasks) * n_trials}")
    print(f"GPUs: {gpus}")
    print(f"Output dir: {output_dir}")

    manager = mp.Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()

    for task in tasks:
        task_queue.put(task)

    processes = []
    for gpu_id in gpus:
        p = mp.Process(
            target=gpu_worker,
            args=(task_queue, gpu_id, seed, output_dir, overwrite, results_queue)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect results
    all_results = []
    while not results_queue.empty():
        all_results.append(results_queue.get())

    # Save JSON summary
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = output_dir / f"optuna_summary_{timestamp_str}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[INFO] JSON summary saved to {summary_path}")

    # Save CSV summary
    csv_path = output_dir / f"optuna_summary_{timestamp_str}.csv"
    with open(csv_path, "w") as f:
        f.write(
            "model,dataset,best_auc,embedding_dim,ffn_out_dim,net_dropout,"
            "net_regularizer,embedding_regularizer\n"
        )
        for r in sorted(all_results, key=lambda x: (x["model"], x["dataset"])):
            bp = r["best_params"]
            f.write(
                f"{r['model']},{r['dataset']},{r['best_value']},"
                f"{bp['embedding_dim']},{bp['ffn_out_dim']},{bp['net_dropout']},"
                f"{bp['net_regularizer']},{bp['embedding_regularizer']}\n"
            )
    print(f"[INFO] CSV summary saved to {csv_path}")

    # Print best configs
    print("\n" + "=" * 70)
    print(" BEST CONFIGURATIONS PER MODEL/DATASET (by AUC) ")
    print("=" * 70)
    for r in sorted(all_results, key=lambda x: (x["model"], x["dataset"])):
        print(f"\n>> {r['model']} / {r['dataset']}")
        print(f"   Best AUC: {r['best_value']:.6f}")
        print(f"   Params: {r['best_params']}")
        print(f"   Trial #: {r['best_trial_number']}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for FuxiCTR (multi-GPU, per-model-dataset study)"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=list(MODEL_CLASS_MAP.keys()),
        help="Models to search (each gets its own study per dataset)"
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        choices=[
            "frappe_x1", "movielenslatest_x1", "kuaivideo_x1",
            "taobaoad_x1", "microvideo1.7m_x1"
        ],
        help="Dataset IDs to search"
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[0],
        help="List of GPU indices, e.g. 0 1 2 3"
    )
    parser.add_argument(
        "--n_trials", type=int, default=30,
        help="Number of Optuna trials per (model, dataset) study"
    )
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="Base random seed"
    )

    # Search space overrides
    parser.add_argument("--embedding_dims", nargs="+", type=int, default=None)
    parser.add_argument("--ffn_out_dims", nargs="+", type=int, default=None)
    parser.add_argument("--net_regularizers", nargs="+", type=float, default=None)
    parser.add_argument("--embedding_regularizers", nargs="+", type=float, default=None)

    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save search results"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing journal logs and start fresh studies"
    )

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
        n_trials=args.n_trials,
        gpus=args.gpus,
        seed=args.seed,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
