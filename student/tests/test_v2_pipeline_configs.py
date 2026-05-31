from __future__ import annotations

import torch.nn as nn

from student.config_utils import load_config, parse_granularities
from student.data.eval_sampling import is_full_scene_sampling, resolve_eval_sampling_config


CONFIGS = [
    "configs/scannet_full_continuous_v2_pseudo_pretrain.yaml",
    "configs/scannet_full_continuous_v2_pseudo_pretrain_g08_binary.yaml",
    "configs/scannet_full_continuous_v2_eval_gt_classagnostic.yaml",
    "configs/scannet_full_continuous_v2_ft_gt_scannet20_classagnostic.yaml",
    "configs/scannet_full_continuous_v2_ft_eval_gt_classagnostic.yaml",
]


def test_v2_pipeline_configs_load_and_use_required_architecture() -> None:
    for path in CONFIGS:
        cfg = load_config(path)
        assert cfg["model"]["decoder_type"] == "continuous_v2"
        assert cfg["model"]["backbone"]["litept_variant"] == "litept_s_star"
        assert cfg["model"]["backbone"]["in_channels"] == 9
        assert cfg["model"]["backbone"]["multi_scale"] is True
        assert cfg["model"]["backbone"]["multi_scale_indices"] == [0, 1, 2, 3]
        assert cfg["model"].get("class_aware_instance", False) is False
        assert cfg["model"].get("allow_partial_decoder_load", False) is False
        assert cfg["eval"]["score_threshold"] == 0.0
        assert is_full_scene_sampling(resolve_eval_sampling_config(cfg["data"], cfg["eval"]))


def test_pretrain_and_finetune_configs_have_expected_sources_and_metrics() -> None:
    pretrain = load_config("configs/scannet_full_continuous_v2_pseudo_pretrain.yaml")
    assert pretrain["data"]["label_source"] == "pack"
    assert parse_granularities(pretrain["data"]) == ("g02", "g05", "g08")
    assert pretrain["train"]["best_val_metric"] == "real_full_scene_official_AP50_scannet20"

    ft = load_config("configs/scannet_full_continuous_v2_ft_gt_scannet20_classagnostic.yaml")
    assert ft["data"]["label_source"] == "scannet_gt"
    assert parse_granularities(ft["data"]) == ("g05",)
    assert ft["data"]["scannet_gt_supervise_all_points"] is True
    assert ft["train"]["prompt_finetune"]["enabled"] is True
    assert ft["train"]["prompt_finetune"]["init_g"] == 0.5
    assert ft["train"]["lr"] == 5.0e-5


def test_g08_binary_pretrain_config_is_score_ablation_only() -> None:
    cfg = load_config("configs/scannet_full_continuous_v2_pseudo_pretrain_g08_binary.yaml")

    assert cfg["experiment"]["name"] == "scannet_full_continuous_v2_pseudo_pretrain_g08_binary"
    assert parse_granularities(cfg["data"]) == ("g08",)
    assert cfg["loss"]["score_target_mode"] == "binary"
    assert cfg["loss"]["score_unmatched_target_mode"] == "zero"
    assert cfg["loss"]["score_loss_balance_mode"] == "pos_neg_balanced"
    assert cfg["loss"]["score_pos_weight"] == 2.0
    assert cfg["loss"]["score_neg_weight"] == 0.5
    assert cfg["loss"]["granularity_weights"] == {"g08": 1.0}
    assert cfg["loss"]["bce_weight"] == 1.0
    assert cfg["loss"]["dice_weight"] == 1.0
    assert cfg["model"]["num_queries"] == 250
    assert cfg["train"]["max_epochs"] == 30
    assert cfg["train"]["eval_every_epochs"] == 5
    assert cfg["train"]["save_every_epochs"] == 5
    assert cfg["train"]["warmup_epochs"] == 5
    assert cfg["debug"]["rich_snapshots"]["granularities"] == ["g08"]
    assert cfg["debug"]["micro_eval"]["granularities"] == ["g08"]


def test_eval_checkpoint_build_model_uses_continuous_v2(monkeypatch) -> None:
    from scripts import eval_checkpoint

    captured: dict = {}

    class FakeModel(nn.Module):
        pass

    def fake_build_student_model(**kwargs):
        captured.update(kwargs)
        return FakeModel()

    cfg = load_config("configs/scannet_full_continuous_v2_eval_gt_classagnostic.yaml")
    monkeypatch.setattr(eval_checkpoint, "build_student_model", fake_build_student_model)

    model = eval_checkpoint._build_model(cfg, parse_granularities(cfg["data"]), "cpu")

    assert isinstance(model, FakeModel)
    assert captured["decoder_type"] == "continuous_v2"
    assert captured["litept_variant"] == "litept_s_star"
    assert captured["in_channels"] == 9
    assert captured["multi_scale"] is True
