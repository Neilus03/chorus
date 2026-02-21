import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

# --- CHORUS PoC CONFIG ---
SCENE_DIR = "scene0000_00"
FRAME_SKIP = 50 # 5578 total frames
DEVICE = "cuda:0"
OUTPUT_DIR = os.path.join(SCENE_DIR, "sam_masks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The official ScanNet20 classes and their corresponding NYUv40 IDs
SCANNET20_MAPPING = {
    "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5,
    "sofa": 6, "table": 7, "door": 8, "window": 9, "bookshelf": 10,
    "picture": 11, "counter": 12, "desk": 14, "curtain": 16,
    "refrigerator": 24, "shower curtain": 28, "toilet": 33,
    "sink": 34, "bathtub": 36, "otherfurniture": 39
}
# Extract just the names for Grounding DINO to use as the text prompt
CLASSES = list(SCANNET20_MAPPING.keys())

print(f"Loading Grounded-SAM onto {DEVICE}...")
# 1. Grounding DINO (Bounding Boxes)
dino_id = "IDEA-Research/grounding-dino-base"
dino_processor = AutoProcessor.from_pretrained(dino_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(DEVICE)

# 2. SAM (Dense Masks)
sam_id = "facebook/sam-vit-huge"
sam_processor = SamProcessor.from_pretrained(sam_id)
sam_model = SamModel.from_pretrained(sam_id).to(DEVICE)

color_dir = os.path.join(SCENE_DIR, "color")
# ScanNet frames are named "0.jpg", "1.jpg", etc. Sort them numerically.
all_frames = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
target_frames = all_frames[::FRAME_SKIP]

text_prompt = ". ".join(CLASSES) + "."
print(f"Processing {len(target_frames)} frames...")
for frame_name in target_frames:
    img_path = os.path.join(color_dir, frame_name)
    image = Image.open(img_path).convert("RGB")

    # --- TEACHER INFERENCE ---
    inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, target_sizes=[image.size[::-1]]
    )[0]

    # Filter by confidence score > 0.3
    keep_mask = results["scores"] > 0.3
    boxes = results["boxes"][keep_mask]

    # Future-proofing: use text_labels instead of labels
    label_key = "text_labels" if "text_labels" in results else "labels"
    labels = [results[label_key][i] for i in range(len(results[label_key])) if keep_mask[i]]

    # Initialize empty mask: 0 = Unlabelled
    frame_mask = np.zeros((image.size[1], image.size[0]), dtype=np.int32)

    if len(boxes) > 0:
        sam_inputs = sam_processor(image, input_boxes=[[boxes.cpu().tolist()]], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)

        # post_process_masks returns a list (one for each image in batch). We take [0].
        processed_masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )[0]

        # processed_masks shape is now (num_boxes, 3, original_H, original_W)
        # SAM outputs 3 mask quality levels per box. We take the first one (index 0).
        best_masks = processed_masks[:, 0, :, :].numpy()

        for i, label_str in enumerate(labels):
            class_name = label_str.split()[0]
            if class_name in CLASSES:
                class_idx = CLASSES.index(class_name) + 1 # 1-indexed

                # Ensure the mask is a boolean array ---
                bool_mask = best_masks[i].astype(bool)
                frame_mask[bool_mask] = class_idx # Paint the mask

    # Save output
    save_name = frame_name.replace(".jpg", ".npy")
    np.save(os.path.join(OUTPUT_DIR, save_name), frame_mask)
    print(f"Saved mask for {frame_name}")

print("Teacher inference complete!")