from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from chorus.export.visualization import save_labeled_mesh_ply


def test_save_labeled_mesh_ply_preserves_faces(tmp_path: Path) -> None:
    source_path = tmp_path / "source_mesh.ply"
    out_path = tmp_path / "colored_mesh.ply"

    vertices = np.array(
        [
            (0.0, 0.0, 0.0, 0, 0, 0),
            (1.0, 0.0, 0.0, 0, 0, 0),
            (0.0, 1.0, 0.0, 0, 0, 0),
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    faces = np.array([([0, 1, 2],)], dtype=[("vertex_indices", "i4", (3,))])
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(faces, "face"),
        ],
        text=False,
    ).write(str(source_path))

    save_labeled_mesh_ply(
        source_ply_path=source_path,
        labels=np.array([0, 1, -1], dtype=np.int32),
        out_path=out_path,
    )

    out_ply = PlyData.read(str(out_path))
    assert len(out_ply["vertex"].data) == 3
    assert len(out_ply["face"].data) == 1
    assert tuple(out_ply["face"].data["vertex_indices"][0]) == (0, 1, 2)
