import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--n", type=int, default=60, help="how many keys to print")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)  # supports raw state_dict too

    if not isinstance(sd, dict):
        raise SystemExit(f"Unexpected checkpoint format. Top-level keys: {list(ckpt.keys())[:50]}")

    keys = list(sd.keys())
    keys_sorted = sorted(keys)

    print(f"ckpt: {args.ckpt}")
    print(f"num_keys: {len(keys)}")
    print("top_level_ckpt_keys:", sorted(list(ckpt.keys()))[:30] if isinstance(ckpt, dict) else type(ckpt))

    # quick signature checks for your case
    has_heads = any(k.startswith("decoder.heads.") for k in keys)
    has_initializers = any(k.startswith("decoder.initializers.") for k in keys)
    has_head = any(k.startswith("decoder.head.") for k in keys)
    has_initializer = any(k.startswith("decoder.initializer.") for k in keys)
    has_gran_mlp = any(k.startswith("decoder.granularity_mlp.") for k in keys)
    has_v2_gran_encoder = any(k.startswith("decoder.granularity_encoder.") for k in keys)
    has_v2_scale_selector = any(k.startswith("decoder.scale_selector.") for k in keys)

    print("\nSignature:")
    print("  expects-continuous-ish (decoder.head / decoder.initializer / granularity_mlp):",
          has_head, has_initializer, has_gran_mlp)
    print("  expects-continuous-v2-ish (decoder.granularity_encoder / scale_selector):",
          has_v2_gran_encoder, has_v2_scale_selector)
    print("  has-per-granularity (decoder.heads.g02 / decoder.initializers.g02):",
          has_heads, has_initializers)

    print("\nFirst keys:")
    for k in keys_sorted[: args.n]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else None
        print(f"  {k}  {shape}")

    print("\nDecoder-related keys (first 80):")
    dec = [k for k in keys_sorted if k.startswith("decoder.")]
    for k in dec[:80]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else None
        print(f"  {k}  {shape}")


if __name__ == "__main__":
    main()
