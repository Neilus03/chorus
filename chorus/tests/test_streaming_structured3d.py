from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chorus.orchestrators.streaming import enumerate_structured3d_scene_ids, read_structured3d_scene_ids


def test_read_structured3d_scene_ids_lists_scene_directories(tmp_path: Path) -> None:
    (tmp_path / "scene_00001").mkdir()
    (tmp_path / "scene_00000").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "README.txt").write_text("x", encoding="utf-8")

    ids = read_structured3d_scene_ids(scans_root=tmp_path, scene_list_file=None, max_scenes=None)
    assert ids == ["scene_00000", "scene_00001"]


def test_enumerate_structured3d_scene_ids() -> None:
    assert enumerate_structured3d_scene_ids(0, 3) == ["scene_00000", "scene_00001", "scene_00002"]
    assert enumerate_structured3d_scene_ids(9, 2) == ["scene_00009", "scene_00010"]


def test_read_structured3d_scene_ids_respects_scene_list_file(tmp_path: Path) -> None:
    lst = tmp_path / "scenes.txt"
    lst.write_text("scene_00002\nscene_00001\n", encoding="utf-8")
    ids = read_structured3d_scene_ids(scans_root=tmp_path, scene_list_file=lst, max_scenes=None)
    assert ids == ["scene_00002", "scene_00001"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
