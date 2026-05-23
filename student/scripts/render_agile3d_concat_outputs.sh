#!/bin/bash
set -euo pipefail

ROOT="${1:?usage: $0 <encoder-pca-root> <figure-root>}"
FIG_ROOT="${2:?usage: $0 <encoder-pca-root> <figure-root>}"

SCENES=(
  scene0655_02
  scene0432_01
  scene0355_00
  scene0377_02
  scene0377_01
  scene0494_00
  scene0081_00
  scene0559_02
  scene0575_02
  scene0616_01
)

mkdir -p "${FIG_ROOT}"

for scene in "${SCENES[@]}"; do
  python /cluster/home/nedela/nedela/projects/chorus/student/scripts/render_agile3d_feature_pca_figure.py \
    --scene-dir "${ROOT}/${scene}" \
    --out "${FIG_ROOT}/${scene}_agile3d_concat_l2.png" \
    --plys input_rgb_points.ply E2_projected_points_pca.ply E3_projected_points_pca.ply Eall_concat_l2_projected_points_pca.ply \
    --titles RGB E2 E3 Eall-concat
done

python - "$FIG_ROOT" <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

root = Path(sys.argv[1])
scenes = [
    "scene0655_02",
    "scene0432_01",
    "scene0355_00",
    "scene0377_02",
    "scene0377_01",
    "scene0494_00",
    "scene0081_00",
    "scene0559_02",
    "scene0575_02",
    "scene0616_01",
]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
except Exception:
    font = ImageFont.load_default()

target_w = 900
pad = 24
label_h = 40
imgs = []
for scene in scenes:
    im = Image.open(root / f"{scene}_agile3d_concat_l2.png").convert("RGB")
    im = im.resize((target_w, int(im.height * target_w / im.width)), Image.Resampling.LANCZOS)
    imgs.append((scene, im))

cols = 2
cell_w = target_w
cell_h = max(im.height for _, im in imgs) + label_h
rows = (len(imgs) + cols - 1) // cols
canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), "white")
draw = ImageDraw.Draw(canvas)
for i, (scene, im) in enumerate(imgs):
    r = i // cols
    c = i % cols
    x = pad + c * (cell_w + pad)
    y = pad + r * (cell_h + pad)
    draw.text((x, y), scene, fill=(0, 0, 0), font=font)
    canvas.paste(im, (x, y + label_h))

canvas.save(root / "agile3d_concat_l2_contact_sheet.png", quality=95)
PY

cp "${ROOT}/manifest.json" "${FIG_ROOT}/manifest.json"
if [[ -f /cluster/home/nedela/nedela/projects/chorus/student/outputs/encoder_pca_figures/top10_pretraining_scene_selection.json ]]; then
  cp /cluster/home/nedela/nedela/projects/chorus/student/outputs/encoder_pca_figures/top10_pretraining_scene_selection.json \
    "${FIG_ROOT}/top10_pretraining_scene_selection.json"
fi

mkdir -p "${FIG_ROOT}/plys"
rsync -a --include='*/' --include='*.ply' --exclude='*' "${ROOT}/" "${FIG_ROOT}/plys/"
