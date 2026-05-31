from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from scripts import eval_checkpoint


def test_eval_checkpoint_loads_raw_state_dict_strict(tmp_path) -> None:
    model = nn.Linear(3, 2)
    ckpt_path = tmp_path / "raw.pt"
    torch.save(model.state_dict(), ckpt_path)

    report = eval_checkpoint.load_checkpoint_for_eval(
        nn.Linear(3, 2),
        ckpt_path,
        device="cpu",
        strict=True,
        report_path=tmp_path / "report.json",
    )

    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["strict"] is True


def test_eval_checkpoint_non_strict_writes_load_report(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    ckpt_path = tmp_path / "partial.pt"
    torch.save({"model_state_dict": {"0.weight": model[0].weight.detach().clone()}}, ckpt_path)
    report_path = tmp_path / "checkpoint_load_report.json"

    report = eval_checkpoint.load_checkpoint_for_eval(
        model,
        ckpt_path,
        device="cpu",
        strict=False,
        report_path=report_path,
    )

    assert report_path.exists()
    assert "0.bias" in report["missing_keys"]
    assert "1.weight" in report["missing_keys"]


def test_eval_checkpoint_script_has_no_training_path() -> None:
    source = inspect.getsource(eval_checkpoint)

    assert "MultiSceneTrainer" not in source
    assert ".train(" not in source
    assert "optimizer.step" not in source
