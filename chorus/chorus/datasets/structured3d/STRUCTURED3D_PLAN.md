# Structured3D CHORUS Integration Plan

This plan is written to be **agent-executable**. It specifies the intended directory layout, the ZIP-in-memory reader, the scene preparation pipeline (export 2D frames + fuse a voxelized `.ply`), and the `SceneAdapter` glue code.

---

## Prerequisites: Directory Setup

Define where the extracted/processed scenes will live for CHORUS to consume.

```bash
mkdir -p /scratch2/nedela/structured3d_scans
```

Recommended companion directory (raw ZIPs):

- `/scratch2/nedela/structured3d_raw` contains the Structured3D release ZIPs.
- We will read the ZIPs directly (no full extraction), and write prepared scenes into `/scratch2/nedela/structured3d_scans/<scene_id>/`.

---

## Phase 1: The Reader (`chorus/datasets/structured3d/reader.py`)

Create this file to handle reading images directly from the ZIP files **in memory**. This is adapted from Pointcept’s reader and adds instance extraction.

### Implementation

```python
import os
import zipfile
import numpy as np
import cv2


class Structured3DReader:
    def __init__(self, files):
        if isinstance(files, str):
            files = [files]
        self.readers = [zipfile.ZipFile(f, "r") for f in files]
        self.names_mapper = dict()
        for idx, reader in enumerate(self.readers):
            for name in reader.namelist():
                self.names_mapper[name] = idx

    def filelist(self):
        return list(self.names_mapper.keys())

    def listdir(self, dir_name):
        dir_name = dir_name.lstrip(os.path.sep).rstrip(os.path.sep)
        file_list = list(
            np.unique(
                [
                    f.replace(dir_name + os.path.sep, "", 1).split(os.path.sep)[0]
                    for f in self.filelist()
                    if f.startswith(dir_name + os.path.sep)
                ]
            )
        )
        if "" in file_list:
            file_list.remove("")
        return file_list

    def read(self, file_name):
        split = self.names_mapper[file_name]
        return self.readers[split].read(file_name)

    def read_camera(self, camera_path):
        z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        cam_extr = np.fromstring(self.read(camera_path), dtype=np.float32, sep=" ")
        cam_t = np.matmul(z2y_top_m, cam_extr[:3] / 1000)
        if cam_extr.shape[0] > 3:
            cam_front, cam_up = cam_extr[3:6], cam_extr[6:9]
            cam_n = np.cross(cam_front, cam_up)
            cam_r = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
            cam_r = np.matmul(z2y_top_m, cam_r)
            cam_f = cam_extr[9:11]
        else:
            cam_r = np.eye(3, dtype=np.float32)
            cam_f = None
        return cam_r, cam_t, cam_f

    def read_depth(self, depth_path):
        depth_data = np.frombuffer(self.read(depth_path), np.uint8)
        depth = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
        return depth  # Returns 2D uint16 array

    def read_color(self, color_path):
        color_data = np.frombuffer(self.read(color_path), np.uint8)
        color = cv2.imdecode(color_data, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]  # BGR to RGB
        return color

    def read_instance(self, instance_path):
        # 16-bit instance PNG
        inst_data = np.frombuffer(self.read(instance_path), np.uint8)
        instance = cv2.imdecode(inst_data, cv2.IMREAD_UNCHANGED)
        return instance
```

### Notes / Constraints

- This reader assumes that each ZIP file contains names like `Structured3D/<scene_id>/...`.
- `read_color()` returns **RGB** numpy arrays.
- `read_depth()` returns raw uint16 depth in **millimeters**.
- `read_camera()` returns `(cam_r, cam_t, cam_f)` consistent with the preparation phase below.

---

## Phase 2: The Preparation Script (`chorus/datasets/structured3d/prepare.py`)

This script does the heavy lifting:

- Reads from the ZIPs via `Structured3DReader`.
- Exports 2D frames to disk for CHORUS/UnSAMv2 (`color/`, `depth/`, `pose/`, `intrinsic/`).
- Back-projects depth into a global point cloud.
- Voxel downsamples to 2cm.
- Saves ScanNet-like geometry as `<scene_id>_vh_clean_2.ply` plus `gt_instance_ids.npy`.
- Writes `.prepared` marker for idempotency.

### Implementation

```python
import os
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

from chorus.datasets.structured3d.reader import Structured3DReader

def is_prepared(scene_dir: Path) -> bool:
    return (scene_dir / ".prepared").exists()

def prepare_structured3d_scene(scene_id: str, raw_zips_dir: str, output_scans_root: str):
    scene_dir = Path(output_scans_root) / scene_id
    if is_prepared(scene_dir):
        return

    print(f"Preparing Structured3D scene: {scene_id}")

    # We ONLY load perspective zips to avoid panorama distortions
    zip_files = [
        os.path.join(raw_zips_dir, f)
        for f in os.listdir(raw_zips_dir)
        if "perspective_full" in f and f.endswith(".zip")
    ]
    reader = Structured3DReader(zip_files)

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"

    for d in [color_dir, depth_dir, pose_dir, intrinsic_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Ensure the scene exists in the dataset
    try:
        rooms = reader.listdir(f"Structured3D/{scene_id}/2D_rendering")
    except Exception as e:
        print(f"Could not find {scene_id} in ZIP files. Error: {e}")
        return

    global_coords = []
    global_colors = []
    global_instances = []

    frame_idx = 0
    intrinsic_saved = False

    for room in rooms:
        prsp_path = f"Structured3D/{scene_id}/2D_rendering/{room}/perspective/full"
        try:
            frames = reader.listdir(prsp_path)
        except Exception:
            continue  # Room might not have perspective/full

        for frame in frames:
            try:
                cam_r, cam_t, cam_f = reader.read_camera(f"{prsp_path}/{frame}/camera_pose.txt")
                depth = reader.read_depth(f"{prsp_path}/{frame}/depth.png")
                color = reader.read_color(f"{prsp_path}/{frame}/rgb_rawlight.png")
                instance = reader.read_instance(f"{prsp_path}/{frame}/instance.png")
            except Exception as e:
                print(f"Skipping {scene_id} room {room} frame {frame} due to error: {e}")
                continue

            height, width = depth.shape
            fx, fy = cam_f

            # Construct Intrinsic Matrix K
            K = np.eye(3)
            K[0, 2], K[1, 2] = width / 2.0, height / 2.0
            K[0, 0] = K[0, 2] / np.tan(fx)
            K[1, 1] = K[1, 2] / np.tan(fy)

            # Construct 4x4 C2W Pose Matrix
            c2w = np.eye(4)
            c2w[:3, :3] = cam_r.T
            c2w[:3, 3] = cam_t

            # --- 1. Export 2D Frames for CHORUS UnSAMv2 ---
            cv2.imwrite(str(color_dir / f"{frame_idx}.jpg"), color[..., ::-1])  # RGB back to BGR for cv2
            cv2.imwrite(str(depth_dir / f"{frame_idx}.png"), depth)
            np.savetxt(str(pose_dir / f"{frame_idx}.txt"), c2w)

            if not intrinsic_saved:
                np.savetxt(str(intrinsic_dir / "intrinsic_color.txt"), K)
                np.savetxt(str(intrinsic_dir / "intrinsic_depth.txt"), K)
                intrinsic_saved = True

            # --- 2. Back-project to 3D for the Global Point Cloud ---
            # Valid depth mask (0 is invalid, 65535 is background in some structured3d formats)
            valid_mask = (depth > 0) & (depth < 65535)
            v, u = np.where(valid_mask)
            z = depth[valid_mask] / 1000.0  # to meters

            # (u - cx) * z / fx
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]

            # Camera coords -> World coords
            pts_camera = np.stack([x, y, z], axis=-1)
            # Apply coordinate transformation matrix (Structured3D local convention)
            pts_camera = pts_camera @ np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
            # NOTE: Pointcept uses world = (coord/1000) @ cam_r.T + cam_t.
            # Here `pts_camera` is already in meters, so we apply `cam_r.T`.
            pts_world = (pts_camera @ cam_r.T) + cam_t

            global_coords.append(pts_world)
            global_colors.append(color[valid_mask])
            global_instances.append(instance[valid_mask])

            frame_idx += 1

    if not global_coords:
        print(f"Scene {scene_id} resulted in empty point cloud.")
        return

    # Merge
    all_coords = np.concatenate(global_coords, axis=0).astype(np.float32)
    all_colors = np.concatenate(global_colors, axis=0).astype(np.uint8)
    all_instances = np.concatenate(global_instances, axis=0).astype(np.int32)

    # Flip coordinates to standard ScanNet upright format if necessary
    all_coords = all_coords @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    # --- 3. Voxel Downsample (0.02m) ---
    grid_size = 0.02
    grid_coord = np.floor(all_coords / grid_size).astype(int)
    _, unique_idx = np.unique(grid_coord, axis=0, return_index=True)

    down_coords = all_coords[unique_idx]
    down_colors = all_colors[unique_idx]
    down_instances = all_instances[unique_idx]

    # Background instances (65535) become 0
    down_instances[down_instances == 65535] = 0

    # Save Geometry to .ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(down_coords)
    pcd.colors = o3d.utility.Vector3dVector(down_colors / 255.0)

    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)

    # Save Ground Truth Instances
    np.save(str(scene_dir / "gt_instance_ids.npy"), down_instances)

    # Mark as prepared
    (scene_dir / ".prepared").touch()
    print(f"Finished {scene_id}. Saved {len(down_coords)} points and {frame_idx} frames.")
```

### Output Directory Contract (Prepared Scene)

After preparation, a scene directory must look like:

```text
/scratch2/nedela/structured3d_scans/<scene_id>/
├── color/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── depth/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── pose/
│   ├── 0.txt
│   ├── 1.txt
│   └── ...
├── intrinsic/
│   ├── intrinsic_color.txt
│   └── intrinsic_depth.txt
├── <scene_id>_vh_clean_2.ply
├── gt_instance_ids.npy
└── .prepared
```

---

## Phase 3: The Adapter (`chorus/datasets/structured3d/adapter.py`)

This glues the extracted data to CHORUS by implementing `SceneAdapter`.

### Required module path (must match runner import)

The dataset-agnostic runner (`chorus/scripts/run_scene.py`) imports:

- `from chorus.datasets.structured3d.adapter import Structured3DSceneAdapter`

So the adapter **must** live at:

- `chorus/chorus/datasets/structured3d/adapter.py`

### Implementation

```python
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData

from chorus.common.types import FrameRecord, GeometryRecord, VisibilityConfig
from chorus.datasets.base import SceneAdapter
from chorus.datasets.structured3d.prepare import prepare_structured3d_scene, is_prepared


class Structured3DSceneAdapter(SceneAdapter):
    def __init__(self, scene_root: Path, raw_zips_dir: str = "/scratch2/nedela/structured3d_raw"):
        super().__init__(scene_root=scene_root)
        self.raw_zips_dir = raw_zips_dir

    @property
    def dataset_name(self) -> str:
        return "structured3d"

    def prepare(self) -> None:
        if not is_prepared(self.scene_root):
            # Pass the scene_id, raw ZIP directory, and the scans root (scene_root.parent).
            prepare_structured3d_scene(self.scene_id, self.raw_zips_dir, str(self.scene_root.parent))

    def list_frames(self) -> list[FrameRecord]:
        color_dir = self.scene_root / "color"
        depth_dir = self.scene_root / "depth"
        pose_dir = self.scene_root / "pose"
        intrinsics_path = self.scene_root / "intrinsic" / "intrinsic_color.txt"

        frame_ids = sorted([p.stem for p in color_dir.glob("*.jpg")], key=int)
        frames = []
        for fid in frame_ids:
            frames.append(
                FrameRecord(
                    frame_id=fid,
                    rgb_path=color_dir / f"{fid}.jpg",
                    depth_path=depth_dir / f"{fid}.png",
                    pose_path=pose_dir / f"{fid}.txt",
                    intrinsics_path=intrinsics_path,
                )
            )
        return frames

    def load_rgb(self, frame: FrameRecord) -> np.ndarray:
        return np.array(Image.open(frame.rgb_path).convert("RGB"))

    def load_depth_m(self, frame: FrameRecord) -> np.ndarray:
        depth = np.array(Image.open(frame.depth_path), dtype=np.float32)
        return depth / 1000.0  # structured3d depth is in mm

    def load_pose_c2w(self, frame: FrameRecord) -> np.ndarray:
        return np.loadtxt(frame.pose_path)

    def load_intrinsics(self, frame: FrameRecord) -> np.ndarray:
        return np.loadtxt(frame.intrinsics_path)[:3, :3]

    def load_geometry_points(self) -> np.ndarray:
        mesh_path = self.scene_root / f"{self.scene_id}_vh_clean_2.ply"
        plydata = PlyData.read(str(mesh_path))
        vertex_data = plydata["vertex"].data
        return np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)

    def load_geometry_colors(self) -> np.ndarray | None:
        mesh_path = self.scene_root / f"{self.scene_id}_vh_clean_2.ply"
        plydata = PlyData.read(str(mesh_path))
        vertex_data = plydata["vertex"].data
        colors = np.stack([vertex_data["red"], vertex_data["green"], vertex_data["blue"]], axis=1).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        return colors

    def get_geometry_record(self) -> GeometryRecord:
        return GeometryRecord(
            geometry_path=self.scene_root / f"{self.scene_id}_vh_clean_2.ply",
            geometry_type="mesh_vertices",
        )

    def get_visibility_config(self) -> VisibilityConfig:
        return VisibilityConfig(
            min_depth_m=0.1,
            z_tolerance_m=0.1,
            depth_scale_to_m=1.0,  # Handled in load_depth_m
            depth_aligned_to_rgb=True,
        )

    def load_gt_instance_ids(self) -> np.ndarray | None:
        path = self.scene_root / "gt_instance_ids.npy"
        if path.exists():
            return np.load(path)
        return None
```

---

## Phase 4: Registration & Execution

### Registration

No additional registration is required for `chorus/scripts/run_scene.py` beyond having the adapter module in the required path above. (If you have a dataset registry elsewhere, you can optionally register it there too.)

### Run CHORUS

Execute the scene processing on your first scene to test the pipeline end-to-end:

```bash
python chorus/scripts/run_scene.py --dataset structured3d \
  --scene-dir /scratch2/nedela/structured3d_scans/scene_00000 \
  --structured3d-raw-zips-dir /scratch2/nedela/structured3d_raw
```

### Expected output location (`training_pack/`)

If the run succeeds and training pack export is enabled (default), CHORUS will write:

- `<scene_dir>/training_pack/`

For the command above, that is:

- `/scratch2/nedela/structured3d_scans/scene_00000/training_pack/`

### What Will Happen (Expected Flow)

- The adapter checks if `.prepared` exists.
- If it doesn’t, it opens the `*perspective_full*.zip` files in `/scratch2/nedela/structured3d_raw` (all matching ZIPs in that directory).
- It exports RGB-D frames to disk and back-projects depth to form a dense, 2cm-voxelized `.ply` house point cloud.
- The adapter returns control to CHORUS.
- CHORUS runs UnSAMv2 over the newly extracted `.jpg` files.
- CHORUS projects SAM masks onto the 3D `.ply` point cloud.
- CHORUS saves the resulting `training_pack/` with multi-granularity geometry targets ready for the student model.



### RELATED: ORIGINAL pointcept's pointcept/datasets/preprocessing/structured3d/preprocess_structured3d.py:

```python
"""
Preprocessing Script for Structured3D

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import argparse
import io
import os
import PIL
from PIL import Image
import cv2
import zipfile
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

VALID_CLASS_IDS_25 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    14,
    15,
    16,
    17,
    18,
    19,
    22,
    24,
    25,
    32,
    34,
    35,
    38,
    39,
    40,
)
CLASS_LABELS_25 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "picture",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "ceiling",
    "refrigerator",
    "television",
    "nightstand",
    "sink",
    "lamp",
    "otherstructure",
    "otherfurniture",
    "otherprop",
)


def normal_from_cross_product(points_2d: np.ndarray) -> np.ndarray:
    xyz_points_pad = np.pad(points_2d, ((0, 1), (0, 1), (0, 0)), mode="symmetric")
    xyz_points_ver = (xyz_points_pad[:, :-1, :] - xyz_points_pad[:, 1:, :])[:-1, :, :]
    xyz_points_hor = (xyz_points_pad[:-1, :, :] - xyz_points_pad[1:, :, :])[:, :-1, :]
    xyz_normal = np.cross(xyz_points_hor, xyz_points_ver)
    xyz_dist = np.linalg.norm(xyz_normal, axis=-1, keepdims=True)
    xyz_normal = np.divide(
        xyz_normal, xyz_dist, out=np.zeros_like(xyz_normal), where=xyz_dist != 0
    )
    return xyz_normal


class Structured3DReader:
    def __init__(self, files):
        super().__init__()
        if isinstance(files, str):
            files = [files]
        self.readers = [zipfile.ZipFile(f, "r") for f in files]
        self.names_mapper = dict()
        for idx, reader in enumerate(self.readers):
            for name in reader.namelist():
                self.names_mapper[name] = idx

    def filelist(self):
        return list(self.names_mapper.keys())

    def listdir(self, dir_name):
        dir_name = dir_name.lstrip(os.path.sep).rstrip(os.path.sep)
        file_list = list(
            np.unique(
                [
                    f.replace(dir_name + os.path.sep, "", 1).split(os.path.sep)[0]
                    for f in self.filelist()
                    if f.startswith(dir_name + os.path.sep)
                ]
            )
        )
        if "" in file_list:
            file_list.remove("")
        return file_list

    def read(self, file_name):
        split = self.names_mapper[file_name]
        return self.readers[split].read(file_name)

    def read_camera(self, camera_path):
        z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        cam_extr = np.fromstring(self.read(camera_path), dtype=np.float32, sep=" ")
        cam_t = np.matmul(z2y_top_m, cam_extr[:3] / 1000)
        if cam_extr.shape[0] > 3:
            cam_front, cam_up = cam_extr[3:6], cam_extr[6:9]
            cam_n = np.cross(cam_front, cam_up)
            cam_r = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
            cam_r = np.matmul(z2y_top_m, cam_r)
            cam_f = cam_extr[9:11]
        else:
            cam_r = np.eye(3, dtype=np.float32)
            cam_f = None
        return cam_r, cam_t, cam_f

    def read_depth(self, depth_path):
        depth = cv2.imdecode(
            np.frombuffer(self.read(depth_path), np.uint8), cv2.IMREAD_UNCHANGED
        )[..., np.newaxis]
        depth[depth == 0] = 65535
        return depth

    def read_color(self, color_path):
        color = cv2.imdecode(
            np.frombuffer(self.read(color_path), np.uint8), cv2.IMREAD_UNCHANGED
        )[..., :3][..., ::-1]
        return color

    def read_segment(self, segment_path):
        segment = np.array(PIL.Image.open(io.BytesIO(self.read(segment_path))))[
            ..., np.newaxis
        ]
        return segment


def parse_scene(
    scene,
    dataset_root,
    output_root,
    ignore_index=-1,
    grid_size=None,
    fuse_prsp=True,
    fuse_pano=True,
    vis=False,
):
    assert fuse_prsp or fuse_pano
    reader = Structured3DReader(
        [
            os.path.join(dataset_root, f)
            for f in os.listdir(dataset_root)
            if f.endswith(".zip")
        ]
    )
    scene_id = int(os.path.basename(scene).split("_")[-1])
    if scene_id < 3000:
        split = "train"
    elif 3000 <= scene_id < 3250:
        split = "val"
    else:
        split = "test"

    print(f"Processing: {scene} in {split}")
    rooms = reader.listdir(os.path.join("Structured3D", scene, "2D_rendering"))
    for room in rooms:
        room_path = os.path.join("Structured3D", scene, "2D_rendering", room)
        coord_list = list()
        color_list = list()
        normal_list = list()
        segment_list = list()
        if fuse_prsp:
            prsp_path = os.path.join(room_path, "perspective", "full")
            frames = reader.listdir(prsp_path)

            for frame in frames:
                try:
                    cam_r, cam_t, cam_f = reader.read_camera(
                        os.path.join(prsp_path, frame, "camera_pose.txt")
                    )
                    depth = reader.read_depth(
                        os.path.join(prsp_path, frame, "depth.png")
                    )
                    color = reader.read_color(
                        os.path.join(prsp_path, frame, "rgb_rawlight.png")
                    )
                    segment = reader.read_segment(
                        os.path.join(prsp_path, frame, "semantic.png")
                    )
                except:
                    print(
                        f"Skipping {scene}_room{room}_frame{frame} perspective view due to loading error"
                    )
                else:
                    fx, fy = cam_f
                    height, width = depth.shape[0], depth.shape[1]
                    pixel = np.transpose(np.indices((width, height)), (2, 1, 0))
                    pixel = pixel.reshape((-1, 2))
                    pixel = np.hstack((pixel, np.ones((pixel.shape[0], 1))))
                    k = np.diag([1.0, 1.0, 1.0])

                    k[0, 2] = width / 2
                    k[1, 2] = height / 2

                    k[0, 0] = k[0, 2] / np.tan(fx)
                    k[1, 1] = k[1, 2] / np.tan(fy)
                    coord = (
                        depth.reshape((-1, 1)) * (np.linalg.inv(k) @ pixel.T).T
                    ).reshape(height, width, 3)
                    coord = coord @ np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
                    normal = normal_from_cross_product(coord)

                    # Filtering invalid points
                    view_dist = np.maximum(
                        np.linalg.norm(coord, axis=-1, keepdims=True), float(10e-5)
                    )
                    cosine_dist = np.sum(
                        (coord * normal / view_dist), axis=-1, keepdims=True
                    )
                    cosine_dist = np.abs(cosine_dist)
                    mask = ((cosine_dist > 0.15) & (depth < 65535) & (segment > 0))[
                        ..., 0
                    ].reshape(-1)

                    coord = np.matmul(coord / 1000, cam_r.T) + cam_t
                    normal = normal_from_cross_product(coord)

                    if sum(mask) > 0:
                        coord_list.append(coord.reshape(-1, 3)[mask])
                        color_list.append(color.reshape(-1, 3)[mask])
                        normal_list.append(normal.reshape(-1, 3)[mask])
                        segment_list.append(segment.reshape(-1, 1)[mask])
                    else:
                        print(
                            f"Skipping {scene}_room{room}_frame{frame} perspective view due to all points are filtered out"
                        )

        if fuse_pano:
            pano_path = os.path.join(room_path, "panorama")
            try:
                _, cam_t, _ = reader.read_camera(
                    os.path.join(pano_path, "camera_xyz.txt")
                )
                depth = reader.read_depth(os.path.join(pano_path, "full", "depth.png"))
                color = reader.read_color(
                    os.path.join(pano_path, "full", "rgb_rawlight.png")
                )
                segment = reader.read_segment(
                    os.path.join(pano_path, "full", "semantic.png")
                )
            except:
                print(f"Skipping {scene}_room{room} panorama view due to loading error")
            else:
                p_h, p_w = depth.shape[:2]
                p_a = np.arange(p_w, dtype=np.float32) / p_w * 2 * np.pi - np.pi
                p_b = np.arange(p_h, dtype=np.float32) / p_h * np.pi * -1 + np.pi / 2
                p_a = np.tile(p_a[None], [p_h, 1])[..., np.newaxis]
                p_b = np.tile(p_b[:, None], [1, p_w])[..., np.newaxis]
                p_a_sin, p_a_cos, p_b_sin, p_b_cos = (
                    np.sin(p_a),
                    np.cos(p_a),
                    np.sin(p_b),
                    np.cos(p_b),
                )
                x = depth * p_a_cos * p_b_cos
                y = depth * p_b_sin
                z = depth * p_a_sin * p_b_cos
                coord = np.concatenate([x, y, z], axis=-1) / 1000
                normal = normal_from_cross_product(coord)

                # Filtering invalid points
                view_dist = np.maximum(
                    np.linalg.norm(coord, axis=-1, keepdims=True), float(10e-5)
                )
                cosine_dist = np.sum(
                    (coord * normal / view_dist), axis=-1, keepdims=True
                )
                cosine_dist = np.abs(cosine_dist)
                mask = ((cosine_dist > 0.15) & (depth < 65535) & (segment > 0))[
                    ..., 0
                ].reshape(-1)
                coord = coord + cam_t

                if sum(mask) > 0:
                    coord_list.append(coord.reshape(-1, 3)[mask])
                    color_list.append(color.reshape(-1, 3)[mask])
                    normal_list.append(normal.reshape(-1, 3)[mask])
                    segment_list.append(segment.reshape(-1, 1)[mask])
                else:
                    print(
                        f"Skipping {scene}_room{room} panorama view due to all points are filtered out"
                    )

        if len(coord_list) > 0:
            coord = np.concatenate(coord_list, axis=0)
            coord = coord @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            color = np.concatenate(color_list, axis=0)
            normal = np.concatenate(normal_list, axis=0)
            normal = normal @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            segment = np.concatenate(segment_list, axis=0)
            segment25 = np.ones_like(segment, dtype=np.int64) * ignore_index
            for idx, value in enumerate(VALID_CLASS_IDS_25):
                mask = np.all(segment == value, axis=-1)
                segment25[mask] = idx

            data_dict = dict(
                coord=coord.astype(np.float32),
                color=color.astype(np.uint8),
                normal=normal.astype(np.float32),
                segment=segment25.astype(np.int16),
            )
            # Grid sampling data
            if grid_size is not None:
                grid_coord = np.floor(coord / grid_size).astype(int)
                _, idx = np.unique(grid_coord, axis=0, return_index=True)
                coord = coord[idx]
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key][idx]

            # Save data
            save_path = os.path.join(
                output_root, split, os.path.basename(scene), f"room_{room}"
            )
            os.makedirs(save_path, exist_ok=True)
            for key in data_dict.keys():
                np.save(os.path.join(save_path, f"{key}.npy"), data_dict[key])

            if vis:
                from pointcept.utils.visualization import save_point_cloud

                os.makedirs("./vis", exist_ok=True)
                save_point_cloud(
                    coord, color / 255, f"./vis/{scene}_room{room}_color.ply"
                )
                save_point_cloud(
                    coord, (normal + 1) / 2, f"./vis/{scene}_room{room}_normal.ply"
                )
        else:
            print(f"Skipping {scene}_room{room} due to no valid points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--grid_size", default=None, type=float, help="Grid size for grid sampling."
    )
    parser.add_argument("--ignore_index", default=-1, type=float, help="Ignore index.")
    parser.add_argument(
        "--fuse_prsp", action="store_true", help="Whether fuse perspective view."
    )
    parser.add_argument(
        "--fuse_pano", action="store_true", help="Whether fuse panorama view."
    )
    config = parser.parse_args()

    reader = Structured3DReader(
        [
            os.path.join(config.dataset_root, f)
            for f in os.listdir(config.dataset_root)
            if f.endswith(".zip")
        ]
    )

    scenes_list = reader.listdir("Structured3D")
    scenes_list = sorted(scenes_list)
    os.makedirs(os.path.join(config.output_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(config.output_root, "val"), exist_ok=True)
    os.makedirs(os.path.join(config.output_root, "test"), exist_ok=True)

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            parse_scene,
            scenes_list,
            repeat(config.dataset_root),
            repeat(config.output_root),
            repeat(config.ignore_index),
            repeat(config.grid_size),
            repeat(config.fuse_prsp),
            repeat(config.fuse_pano),
        )
    )
    pool.shutdown()
```