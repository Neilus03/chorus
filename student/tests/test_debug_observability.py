import time

from student.engine.debug_observability import DebugObserver


def test_debug_disabled_no_writer_no_artifacts(tmp_path):
    obs = DebugObserver(
        output_dir=tmp_path,
        debug_config={"enabled": False, "tensorboard": {"enabled": True}},
        is_main_process=True,
    )
    assert obs.writer is None
    assert not (tmp_path / "tensorboard").exists()
    assert not (tmp_path / "debug_snapshots").exists()
    obs.close()


def test_tensorboard_writer_created(tmp_path):
    obs = DebugObserver(
        output_dir=tmp_path,
        debug_config={
            "enabled": True,
            "tensorboard": {"enabled": True, "log_dir_name": "tensorboard", "flush_secs": 1},
            "scalars": {"log_to_tensorboard": True},
        },
        is_main_process=True,
    )
    assert obs.writer is not None
    obs.log_scalars({"debug/loss/total": 1.0}, step=1)
    obs.close()
    # SummaryWriter writes asynchronously; give the filesystem a brief moment.
    for _ in range(20):
        if list((tmp_path / "tensorboard").glob("events.out.tfevents.*")):
            break
        time.sleep(0.05)
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
