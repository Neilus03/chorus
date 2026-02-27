import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _as_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PoC3 config yaml: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}: expected mapping")
    return data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = _as_path(
    os.environ.get("POC3_CONFIG", str(Path(__file__).with_name("config.yaml")))
)
RAW_CONFIG = _load_yaml_config(CONFIG_PATH)

PROJECT_CFG = RAW_CONFIG.get("project", {})
STORAGE_CFG = RAW_CONFIG.get("storage", {})
EXTERNAL_CFG = RAW_CONFIG.get("external", {})
SCRIPTS_CFG = RAW_CONFIG.get("scripts", {})
DEFAULTS_CFG = RAW_CONFIG.get("defaults", {})
WANDB_CONFIG = RAW_CONFIG.get("wandb", {})

POC3_DIR = PROJECT_ROOT / PROJECT_CFG.get("poc3_dirname", "poc3")

SCANNET_ROOT = _as_path(STORAGE_CFG.get("scannet_root", "/scratch2/nedela/chorus_poc"))
SCANS_ROOT = SCANNET_ROOT / STORAGE_CFG.get("scans_subdir", "scans")
REPORTS_ROOT = SCANNET_ROOT / STORAGE_CFG.get("reports_subdir", "reports_poc3")
DEFAULT_SCENE_LIST_FROM_DIR = _as_path(
    STORAGE_CFG.get("scene_list_default_from_dir", "/scratch2/nedela/scannet_processed/train")
)

SCANNET_DOWNLOADER = _as_path(
    EXTERNAL_CFG.get(
        "scannet_downloader", str(SCANNET_ROOT / "tools" / "download-scannet.py")
    )
)
LITEPT_SCANNET_PAIR = _as_path(
    EXTERNAL_CFG.get(
        "litept_scannet_pair",
        "/home/nedela/projects/LitePT/datasets/preprocessing/scannet/scannet_pair",
    )
)

CHORUS_TEACHER_SCRIPT = POC3_DIR / SCRIPTS_CFG.get("chorus_teacher", "chorus_teacher.py")
CHORUS_PROJECT_CLUSTER_SCRIPT = POC3_DIR / SCRIPTS_CFG.get(
    "chorus_project_cluster", "chorus_project_cluster.py"
)
CHORUS_ORACLE_EVAL_SCRIPT = POC3_DIR / SCRIPTS_CFG.get(
    "chorus_oracle_eval", "chorus_oracle_eval.py"
)

DEFAULT_GRANULARITIES = [str(g) for g in DEFAULTS_CFG.get("granularities", ["0.2", "0.5", "0.8"])]
DEFAULT_FRAME_SKIP = int(DEFAULTS_CFG.get("frame_skip", 10))
DEFAULT_KEEP_FULL_MODULO = int(DEFAULTS_CFG.get("keep_full_modulo", 100))
DEFAULT_DELETE_INTERMEDIATE = bool(DEFAULTS_CFG.get("delete_intermediate", True))
DEFAULT_CONTINUE_ON_ERROR = bool(DEFAULTS_CFG.get("continue_on_error", True))

