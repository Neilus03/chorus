#!/usr/bin/env python3
"""Write ``normals.npy`` into CHORUS training packs from ScanNet mesh vertices.

``points.npy`` must match vertex order and count of ``<scene_id>_vh_clean_2.ply``
(standard ScanNet / CHORUS export). Mismatches are rejected with an error.

If the PLY has no ``nx``/``ny``/``nz`` (common for raw ScanNet downloads), normals
are computed from the triangle mesh (Open3D if available, else NumPy).

Examples (paths are resolved from the git repo root that contains ``student/`` and
``chorus/``; adjust ``--scans-root`` if your data live elsewhere):

- Batch: ``python chorus/scripts/add_normals_to_training_packs.py --scene-list
  configs/splits/train_100.txt --scans-root /cluster/work/igp_psr/nedela/chorus_poc/scans``
- One scene: ``python chorus/scripts/add_normals_to_training_packs.py --scene-dir
  /cluster/work/igp_psr/nedela/chorus_poc/scans/scene0000_00``
- Dry-run: add ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
# chorus/scripts -> chorus (pkg) -> repo root (…/chorus with student/ + chorus/)
_GIT_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Keep in sync with ``student.data.training_pack._resolve_pack_dir`` (avoid
# importing ``student.data``, which pulls the full dataset stack).
_PACK_DIR_NAMES = ("training_pack", "litept_pack")


def _resolve_pack_dir(path: Path) -> Path:
    """Scene dir or pack dir → directory containing ``scene_meta.json``."""
    path = Path(path)
    if (path / "scene_meta.json").is_file():
        return path
    for name in _PACK_DIR_NAMES:
        candidate = path / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No training pack found at or under {path} (looked for {_PACK_DIR_NAMES})"
    )


log = logging.getLogger("add_normals")


def _read_scene_dirs(scene_list_file: Path, scans_root: Path) -> list[Path]:
    """Same contract as ``build_scene_list`` (train split file + scans root)."""
    scene_list_file = Path(scene_list_file)
    scans_root = Path(scans_root)
    if not scene_list_file.is_file():
        raise FileNotFoundError(f"Scene list file not found: {scene_list_file}")
    scene_dirs: list[Path] = []
    for line in scene_list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scene_dir = scans_root / line
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        _resolve_pack_dir(scene_dir)
        scene_dirs.append(scene_dir)
    if not scene_dirs:
        raise ValueError(f"No scenes found in {scene_list_file}")
    log.info("Loaded %d scene paths from %s", len(scene_dirs), scene_list_file)
    return scene_dirs


def _find_mesh_ply(scene_dir: Path) -> Path | None:
    """Locate ``<scene_id>_vh_clean_2.ply`` in *scene_dir* (ScanNet convention)."""
    sid = scene_dir.name
    explicit = scene_dir / f"{sid}_vh_clean_2.ply"
    if explicit.is_file():
        return explicit
    cand = sorted(scene_dir.glob("*_vh_clean_2.ply"))
    if cand:
        return cand[0]
    cand = sorted(scene_dir.glob("*.ply"))
    return cand[0] if cand else None


def _vertex_normals_from_stored_nx(ply_vertex_element) -> np.ndarray:
    v = ply_vertex_element.data
    nx = np.asarray(v["nx"], dtype=np.float32)
    ny = np.asarray(v["ny"], dtype=np.float32)
    nz = np.asarray(v["nz"], dtype=np.float32)
    return np.stack([nx, ny, nz], axis=1)


def _vertices_and_faces_from_plydata(ply) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (N,3) float64 and triangle indices (M,3) int64 from PlyData."""
    vx = np.asarray(ply["vertex"]["x"], dtype=np.float64)
    vy = np.asarray(ply["vertex"]["y"], dtype=np.float64)
    vz = np.asarray(ply["vertex"]["z"], dtype=np.float64)
    verts = np.stack([vx, vy, vz], axis=1)
    face_el = next((e for e in ply.elements if e.name == "face"), None)
    if face_el is None:
        return verts, np.empty((0, 3), dtype=np.int64)
    fdata = face_el.data
    names = fdata.dtype.names or ()
    key = "vertex_indices" if "vertex_indices" in names else names[0] if names else None
    if key is None:
        return verts, np.empty((0, 3), dtype=np.int64)
    vi = fdata[key]
    if isinstance(vi, np.ndarray) and vi.ndim == 2 and vi.shape[1] == 3:
        return verts, vi.astype(np.int64, copy=False)
    rows: list[np.ndarray] = []
    for row in vi:
        a = np.asarray(row, dtype=np.int64).ravel()
        if a.size == 3:
            rows.append(a)
        elif a.size > 3:
            # Fan triangulation for n-gons (unlikely for ScanNet vh_clean_2).
            for k in range(1, a.size - 1):
                rows.append(np.array([a[0], a[k], a[k + 1]], dtype=np.int64))
    if not rows:
        return verts, np.empty((0, 3), dtype=np.int64)
    return verts, np.stack(rows, axis=0)


def _vertex_normals_from_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Uniform face-normal accumulation per vertex (no Open3D)."""
    if faces.shape[0] == 0:
        raise ValueError("No faces in mesh; cannot estimate normals")
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(fn, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-12)
    fn = fn / lens
    vn_acc = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(vn_acc, faces[:, j], fn)
    raw_len = np.linalg.norm(vn_acc, axis=1)
    lens_v = np.maximum(raw_len[:, np.newaxis], 1e-12)
    out = (vn_acc / lens_v).astype(np.float32)
    bad = raw_len < 1e-9
    if np.any(bad):
        out = out.copy()
        out[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return out


def _vertex_normals_from_ply(ply_path: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v_el = ply["vertex"]
    v = v_el.data
    n = len(v)
    if n == 0:
        raise ValueError(f"Empty vertex element: {ply_path}")
    names = v.dtype.names or ()
    if all(x in names for x in ("nx", "ny", "nz")):
        return _vertex_normals_from_stored_nx(v_el)

    log.info(
        "[%s] ply has no nx/ny/nz (%s); computing vertex normals from geometry",
        ply_path.name,
        names,
    )

    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            out = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if out.shape[0] == n:
                return out
            log.warning(
                "open3d vertex count %d != ply vertex count %d; falling back to numpy",
                out.shape[0],
                n,
            )
    except Exception as ex:
        log.warning("open3d normal estimation failed (%s); using numpy", ex)

    verts, faces = _vertices_and_faces_from_plydata(ply)
    if verts.shape[0] != n:
        raise ValueError(
            f"Vertex count mismatch after ply parse: {verts.shape[0]} vs {n}"
        )
    return _vertex_normals_from_triangles(verts, faces)


def _write_normals_for_pack(
    scene_dir: Path,
    *,
    dry_run: bool,
) -> bool:
    pack_dir = _resolve_pack_dir(scene_dir)
    points_path = pack_dir / "points.npy"
    if not points_path.exists():
        log.warning("[%s] no points.npy in %s", scene_dir.name, pack_dir)
        return False

    pts = np.load(points_path)
    n_pts = int(pts.shape[0])

    meta_path = pack_dir / "scene_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_meta = meta.get("num_points")
        if n_meta is not None and int(n_meta) != n_pts:
            log.error(
                "[%s] num_points in scene_meta (%s) != points.npy rows (%d)",
                scene_dir.name,
                n_meta,
                n_pts,
            )
            return False

    mesh_path = _find_mesh_ply(scene_dir)
    if mesh_path is None:
        log.warning("[%s] no ScanNet ply found in %s", scene_dir.name, scene_dir)
        return False

    normals = _vertex_normals_from_ply(mesh_path)
    if normals.shape[0] != n_pts:
        log.error(
            "[%s] vertex count mismatch: points.npy N=%d mesh N=%d (%s)",
            scene_dir.name,
            n_pts,
            normals.shape[0],
            mesh_path.name,
        )
        return False

    out_path = pack_dir / "normals.npy"
    if dry_run:
        log.info("[dry-run] would write %s  shape=%s", out_path, normals.shape)
        return True

    np.save(out_path, normals.astype(np.float32, copy=False))

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        opt = meta.setdefault("optional_files_present", {})
        opt["normals.npy"] = True
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    log.info("[%s] wrote %s  (%d, 3)", scene_dir.name, out_path, n_pts)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=str,
        default=None,
        help="Single ScanNet scene directory (contains training_pack or mesh ply).",
    )
    parser.add_argument(
        "--scene-list",
        type=str,
        default=None,
        help="Text file of scene ids (same as training).",
    )
    parser.add_argument(
        "--scans-root",
        type=str,
        default=None,
        help="Root for scene dirs when using --scene-list.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without writing normals.npy",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "With --scene-list: do not exit non-zero when some scenes fail; "
            "exit 1 only if every scene failed."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.scene_dir:
        ok = _write_normals_for_pack(Path(args.scene_dir), dry_run=args.dry_run)
        raise SystemExit(0 if ok else 1)

    if args.scene_list and args.scans_root:
        list_path = Path(args.scene_list)
        if not list_path.is_file():
            list_path = _GIT_REPO_ROOT / "student" / args.scene_list
        dirs = _read_scene_dirs(list_path, Path(args.scans_root))
        n_ok = 0
        failed: list[str] = []
        for d in dirs:
            if _write_normals_for_pack(d, dry_run=args.dry_run):
                n_ok += 1
            else:
                failed.append(d.name)
        log.info("Done: %d / %d scenes", n_ok, len(dirs))
        if failed:
            preview = ", ".join(failed[:30])
            if len(failed) > 30:
                preview += ", ..."
            log.warning("Failed %d scene(s): %s", len(failed), preview)
        if args.continue_on_error:
            raise SystemExit(0 if n_ok > 0 else 1)
        raise SystemExit(0 if n_ok == len(dirs) else 1)

    parser.error("Provide --scene-dir or both --scene-list and --scans-root")


if __name__ == "__main__":
    main()
