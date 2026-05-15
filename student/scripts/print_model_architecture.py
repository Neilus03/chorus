#!/usr/bin/env python3
"""Dump the full student model architecture from a YAML config.

This intentionally avoids dataset/trainer construction so it can be used as a
quick architecture inspection tool on the cluster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
STUDENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = STUDENT_ROOT.parent
if str(STUDENT_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDENT_ROOT))

from student.config_utils import load_config, parse_granularities, resolve_num_queries
from student.models.finetune_wrapper import FineTuningWrapper
from student.models.student_model import build_student_model


def _normalise_config_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute():
        for root in (STUDENT_ROOT, REPO_ROOT):
            candidate = root / p
            if candidate.exists():
                return candidate
    return p


def build_from_config(config_path: Path):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    bb_cfg = model_cfg["backbone"]

    granularities = parse_granularities(data_cfg)
    num_queries, num_queries_by_granularity = resolve_num_queries(model_cfg, bb_cfg)
    decoder_type = str(model_cfg.get("decoder_type", "multi_head"))

    model = build_student_model(
        litept_root=bb_cfg["litept_root"],
        in_channels=bb_cfg.get("in_channels", 3),
        grid_size=bb_cfg.get("grid_size", 0.02),
        litept_variant=bb_cfg.get("litept_variant", "litept_s_star"),
        litept_kwargs=bb_cfg.get("litept_kwargs", None),
        hidden_dim=model_cfg.get("decoder_hidden_dim", 256),
        num_queries=num_queries,
        num_queries_by_granularity=num_queries_by_granularity,
        granularities=granularities,
        num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
        num_decoder_heads=model_cfg.get("num_decoder_heads", 8),
        query_init=model_cfg.get("query_init", "hybrid"),
        use_positional_guidance=model_cfg.get("use_positional_guidance", True),
        learned_query_ratio=model_cfg.get("learned_query_ratio", 0.25),
        multi_scale=bb_cfg.get("multi_scale", False),
        multi_scale_indices=bb_cfg.get("multi_scale_indices", None),
        decoder_type=decoder_type,
        num_instance_classes=(
            int(model_cfg["num_instance_classes"])
            if bool(model_cfg.get("class_aware_instance", False))
            else None
        ),
    )

    prompt_ft_cfg = train_cfg.get("prompt_finetune", {})
    if isinstance(prompt_ft_cfg, bool):
        prompt_ft_enabled = prompt_ft_cfg
        prompt_ft_cfg = {"enabled": prompt_ft_enabled}
    elif isinstance(prompt_ft_cfg, dict):
        prompt_ft_enabled = bool(prompt_ft_cfg.get("enabled", False))
    else:
        prompt_ft_enabled = False

    if prompt_ft_enabled:
        model = FineTuningWrapper(
            model,
            init_g=float(prompt_ft_cfg.get("init_g", 0.5)),
            backbone_lr_scale=float(
                prompt_ft_cfg.get("backbone_lr_scale", train_cfg.get("backbone_lr_scale", 0.01))
            ),
            mode=str(prompt_ft_cfg.get("mode", "learned")),
        )

    return model, cfg, granularities


def write_parameter_table(model, out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"{'parameter':90s} {'shape':28s} {'numel':>14s} trainable")
    lines.append("-" * 145)
    for name, param in model.named_parameters():
        shape = "x".join(str(v) for v in param.shape) or "scalar"
        lines.append(f"{name:90s} {shape:28s} {param.numel():14,d} {param.requires_grad}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append("-" * 145)
    lines.append(f"{'TOTAL':90s} {'':28s} {total:14,d}")
    lines.append(f"{'TRAINABLE':90s} {'':28s} {trainable:14,d}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config path, e.g. configs/scannet_full_continuous.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to student/logs/model_architecture/<config-stem>",
    )
    args = parser.parse_args()

    config_path = _normalise_config_path(args.config)
    out_dir = args.out or (STUDENT_ROOT / "logs" / "model_architecture" / config_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading config and constructing model from {config_path}", flush=True)
    model, cfg, granularities = build_from_config(config_path)
    print("model constructed; writing architecture repr", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    (out_dir / "architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
    print("writing parameter table", flush=True)
    write_parameter_table(model, out_dir / "parameters.txt")
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"config: {config_path}",
                f"experiment: {cfg.get('experiment', {}).get('name', '<unknown>')}",
                f"granularities: {granularities}",
                f"total_params: {total_params:,}",
                f"trainable_params: {trainable_params:,}",
                f"architecture: {out_dir / 'architecture.txt'}",
                f"parameters: {out_dir / 'parameters.txt'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print((out_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
