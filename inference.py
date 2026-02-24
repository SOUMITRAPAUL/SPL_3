"""
Inference — enhance a single image or a folder of images.

Usage:
  python inference.py --input dark_photo.jpg --output result.png \
                      --checkpoint checkpoints/best.pth

  python inference.py --input images/low/ --output images/enhanced/ \
                      --checkpoint checkpoints/best.pth

Post-processing sliders (applied AFTER model):
  --enhance    1.0   blend between original and enhanced (0=original, 1=full)
  --brightness 1.0   PIL brightness multiplier
  --contrast   1.0   PIL contrast multiplier
"""

import argparse
import os
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

try:
    from skimage.restoration import denoise_tv_chambolle
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Denoising will be disabled.")

# Deferred imports in functions
# from model import DRSformer
# from model.alignment import apply_alignment

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path, base_ch=32, num_dmrb=3, device="cpu"):
    """
    Load DRSformer model. Handles both raw state_dicts and wrapped checkpoints.
    Support checkpoints from LOL-v1, LOL-v2, and DataParallel training.
    """
    print(f"  [load_model] STEP 4.1: torch.load from {checkpoint_path}", flush=True)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        elif k.startswith("stage1."):
            new_state[k.replace("stage1.", "body.0.")] = v
        elif k.startswith("stage2."):
            new_state[k.replace("stage2.", "body.1.")] = v
        else:
            new_state[k] = v

    print(f"  [load_model] STEP 4.2: Instantiating DRSformer", flush=True)
    from model import DRSformer
    model = DRSformer(dim=base_ch).to(device)
    
    print(f"  [load_model] STEP 4.3: Loading state dict", flush=True)
    try:
        model.load_state_dict(new_state)
    except RuntimeError as e:
        print(f"  [Warning] Strict load failed: {e}", flush=True)
        model.load_state_dict(new_state, strict=False)

    model.eval()
    print(f"  [load_model] STEP 4.4: Model ready", flush=True)
    return model


def pad_to_multiple(tensor, multiple=16):
    """Pad H and W so they are divisible by `multiple` (required by deep architectures)."""
    _, _, h, w = tensor.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        tensor = F.pad(tensor, (0, pw, 0, ph), mode="reflect")
    return tensor, h, w


@torch.no_grad()
def enhance_image(model, pil_img,
                  enhance_strength: float = 1.0,
                  brightness: float = 1.0,
                  contrast:   float = 1.0,
                  gamma:      float = 1.0,
                  saturation: float = 1.0,
                  sharpness:  float = 1.0,
                  denoise_weight: float = 0.00,
                  auto_align: bool = False,
                  device="cpu"):
    """
    Args:
        pil_img          : RGB PIL Image
        enhance_strength : 0.0 = original, 1.0 = full model output
        brightness       : PIL brightness multiplier (post-processing)
        contrast         : PIL contrast multiplier   (post-processing)
        gamma            : Power-law transform (1.0 = neutral, >1.0 = richer shadows)
        saturation       : PIL color multiplier
        sharpness        : PIL sharpness multiplier
        denoise_weight   : Weight for TV denoising (0.0 = off)
    Returns:
        Enhanced RGB PIL Image
    """
    if denoise_weight > 0:
        if HAS_SKIMAGE:
            # Convert to numpy for TV denoising (subtle cleaning)
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            # A weight of 0.02 is much more conservative than before
            clean_np = denoise_tv_chambolle(img_np, weight=denoise_weight, channel_axis=-1)
            pil_img = Image.fromarray((clean_np * 255.0).astype(np.uint8))
        else:
            print("Denoising skipped: scikit-image not installed.")

    # Domain Alignment (Camera Sensor Independence)
    if auto_align:
        from model.alignment import apply_alignment
        pil_img = apply_alignment(pil_img)


    inp = TF.to_tensor(pil_img).unsqueeze(0).to(device)   # [1,3,H,W]

    inp_pad, orig_h, orig_w = pad_to_multiple(inp, multiple=16)

    out_pad = model(inp_pad)
    out     = out_pad[:, :, :orig_h, :orig_w]             # crop padding

    # Blend with original (enhance strength slider)
    if enhance_strength < 1.0:
        orig_crop = inp[:, :, :orig_h, :orig_w]
        out = orig_crop * (1.0 - enhance_strength) + out * enhance_strength

    result = TF.to_pil_image(out.squeeze(0).clamp(0, 1).cpu())

    # Post-processing (user preference adjustments)
    if brightness != 1.0:
        result = ImageEnhance.Brightness(result).enhance(brightness)
    if contrast != 1.0:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    if saturation != 1.0:
        result = ImageEnhance.Color(result).enhance(saturation)
    if sharpness != 1.0:
        result = ImageEnhance.Sharpness(result).enhance(sharpness)
    
    # Advanced Shadow Management (Gamma)
    if gamma != 1.0:
        # 1.0/gamma because usually "higher gamma slider" means "make darks more visible"
        # Using a lookup table (point) for speed
        inv_gamma = 1.0 / gamma
        table = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
        result = result.point(table * 3)  # Apply to all 3 channels

    return result


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MSPFN inference")
    p.add_argument("--input",      type=str, required=True,
                   help="Input image or folder")
    p.add_argument("--output",     type=str, required=True,
                   help="Output image or folder")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    p.add_argument("--base_ch",    type=int,   default=32)
    p.add_argument("--num_dmrb",   type=int,   default=3)
    # User-adjustable sliders
    p.add_argument("--enhance",    type=float, default=1.0,
                   help="Enhancement strength: 0=original, 1=fully enhanced")
    p.add_argument("--brightness", type=float, default=1.0,
                   help="Brightness multiplier (e.g. 1.2 = 20%% brighter)")
    p.add_argument("--contrast",   type=float, default=1.0,
                   help="Contrast multiplier")
    p.add_argument("--align", action="store_true",
                   help="Enable Auto-Alignment (White Balance + Exposure)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, args.base_ch, args.num_dmrb, device)
    print(f"Model loaded. Device: {device}")

    inp_path = Path(args.input)
    out_path = Path(args.output)

    if inp_path.is_file():
        files      = [inp_path]
        out_is_dir = False
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        files      = sorted(p for p in inp_path.iterdir()
                             if p.suffix.lower() in IMG_EXTS)
        out_is_dir = True
        out_path.mkdir(parents=True, exist_ok=True)

    print(f"Enhancing {len(files)} image(s) ...")
    for img_path in files:
        pil = Image.open(img_path).convert("RGB")
        res = enhance_image(model, pil,
                            enhance_strength=args.enhance,
                            brightness=args.brightness,
                            contrast=args.contrast,
                            auto_align=args.align,
                            device=device)
        save_path = (out_path / (img_path.stem + "_enhanced.png")
                     if out_is_dir else out_path)
        res.save(save_path)
        print(f"  {img_path.name}  →  {save_path}")

    print("Done.")


if __name__ == "__main__":
    main()
