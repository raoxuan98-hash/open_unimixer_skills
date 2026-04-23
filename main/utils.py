import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FUXICTR_ROOT = PROJECT_ROOT / "FuxiCTR"
DATA_ROOT = PROJECT_ROOT / "data"
UNIFIED_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset_config.yaml"

# Ensure main/ is on path so run_expid can be imported when utils is used externally
_MAIN_DIR = Path(__file__).parent.resolve()
if str(_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_MAIN_DIR))

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
}


def fix_config_paths():
    """修复统一的 dataset_config.yaml 中的旧绝对路径为当前环境的正确路径."""
    if not UNIFIED_DATASET_CONFIG.exists():
        return

    with open(UNIFIED_DATASET_CONFIG, "r") as f:
        content = f.read()

    # 如果已经包含当前项目根目录的路径，不需要修复
    if str(PROJECT_ROOT) in content:
        return

    # 替换旧的 Mac 绝对路径为当前环境的正确路径
    old_prefix = "/Users/raoxuan/kuaishou/unimixer/unimixer_open_datasets/data"
    if old_prefix in content:
        content = content.replace(old_prefix, str(DATA_ROOT))
        with open(UNIFIED_DATASET_CONFIG, "w") as f:
            f.write(content)
        print(f"[INFO] Fixed paths in {UNIFIED_DATASET_CONFIG}")
        return

    # 兜底：匹配 /Users/*/.../data 或类似绝对路径
    modified = False
    patterns = [
        r"/Users/[^/\n]+/[^/\n]+/unimixer_open_datasets/data",
        r"/Users/[^/\n]+/[^/\n]+/[^/\n]+/unimixer_open_datasets/data",
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, str(DATA_ROOT), content)
            modified = True
    if modified:
        with open(UNIFIED_DATASET_CONFIG, "w") as f:
            f.write(content)
        print(f"[INFO] Fixed paths in {UNIFIED_DATASET_CONFIG}")


def run_model(model_name, dataset_id, gpu=0):
    if model_name not in MODEL_CLASS_MAP:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"Supported: {', '.join(MODEL_CLASS_MAP.keys())}")
        return 1

    class_name = MODEL_CLASS_MAP[model_name]
    exp_id = f"{class_name}_{dataset_id}_unified"
    model_dir = FUXICTR_ROOT / "model_zoo" / model_name
    config_dir = model_dir / "config"

    if not config_dir.exists():
        print(f"[ERROR] Config directory not found: {config_dir}")
        return 1

    fix_config_paths()

    # Set GPU environment
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        fuxi_gpu = 0  # After isolation, FuxiCTR sees only one GPU at index 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        fuxi_gpu = -1

    print(f"\n{'='*60}")
    print(f"Running {model_name} on {dataset_id}")
    print(f"ExpID: {exp_id}")
    print(f"GPU: {gpu if gpu >= 0 else 'CPU'}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"FuxiCTR internal --gpu={fuxi_gpu}")
    print(f"{'='*60}")

    # Import and run the unified training entry
    import run_expid
    return run_expid.run_experiment(model_name, exp_id, fuxi_gpu)
