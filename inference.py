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

# Heavy imports are deferred inside load_model and enhance_image
# to prevent Gunicorn timeout during the initial module load.

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
class EnhancementModel:
    """
    CRC Class: EnhancementModel
    Responsibilities: Loading deep learning models, performing image enhancement.
    """
    def __init__(self, weightsPath=None, device="cpu"):
        self.weightsPath = weightsPath
        self.device = device
        self.model = None
        self.inputShape = None

    def loadModel(self, checkpoint_path=None, base_ch=32, device=None):
        """
        Equivalent to CRC loadModel()
        """
        import torch
        if checkpoint_path:
            self.weightsPath = checkpoint_path
        if device:
            self.device = device
            
        print(f"  [EnhancementModel] Loading from {self.weightsPath}", flush=True)
        ckpt  = torch.load(self.weightsPath, map_location=self.device, weights_only=False)
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

        from model import DRSformer
        self.model = DRSformer(dim=base_ch).to(self.device)
        
        try:
            self.model.load_state_dict(new_state)
        except RuntimeError:
            self.model.load_state_dict(new_state, strict=False)

        self.model.eval()
        print(f"  [EnhancementModel] Model ready", flush=True)
        return self.model

    def enhanceImage(self, lowLightImage, userPreferences, device=None):
        """
        Equivalent to CRC enhanceImage(lowLightImage, userPreferences)
        """
        import torch
        import numpy as np
        import torchvision.transforms.functional as TF
        from PIL import Image, ImageEnhance
        
        target_device = device if device else self.device
        
        try:
            from skimage.restoration import denoise_tv_chambolle
            HAS_SKIMAGE = True
        except ImportError:
            HAS_SKIMAGE = False

        pil_img = lowLightImage.data
        
        # Prefs
        enhance_strength = getattr(userPreferences, 'enhance_strength', 1.0)
        brightness       = getattr(userPreferences, 'brightness', 1.0)
        contrast         = getattr(userPreferences, 'contrast', 1.0)
        gamma            = getattr(userPreferences, 'gamma', 1.0)
        saturation       = getattr(userPreferences, 'saturation', 1.0)
        sharpness        = getattr(userPreferences, 'sharpness', 1.0)
        denoise_weight   = getattr(userPreferences, 'denoise_weight', 0.0)
        auto_align       = getattr(userPreferences, 'auto_align', False)

        if denoise_weight > 0 and HAS_SKIMAGE:
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            clean_np = denoise_tv_chambolle(img_np, weight=denoise_weight, channel_axis=-1)
            pil_img = Image.fromarray((clean_np * 255.0).astype(np.uint8))
            del img_np, clean_np

        if auto_align:
            from model.alignment import apply_alignment
            prev_pil = pil_img
            pil_img = apply_alignment(pil_img)
            # Ensure the previous image is closed if it was opened from file
            if prev_pil != lowLightImage.data:
                prev_pil.close()

        import gc
        with torch.no_grad():
            inp = TF.to_tensor(pil_img).unsqueeze(0).to(target_device)
            inp_pad, orig_h, orig_w = pad_to_multiple(inp, multiple=16)

            out_pad = self.model(inp_pad)
            
            # Crop to original size
            out = out_pad[:, :, :orig_h, :orig_w]
            del out_pad, inp_pad

            if enhance_strength < 1.0:
                orig_crop = inp[:, :, :orig_h, :orig_w]
                out = orig_crop * (1.0 - enhance_strength) + out * enhance_strength
                del orig_crop

            result = TF.to_pil_image(out.squeeze(0).clamp(0, 1).cpu())
            
            # Cleanup tensors
            del inp, out
            
            # Check if device is CUDA (handle both device object and string)
            is_cuda = False
            if hasattr(target_device, 'type'):
                is_cuda = (target_device.type == 'cuda')
            elif isinstance(target_device, str):
                is_cuda = ('cuda' in target_device)
                
            if is_cuda:
                torch.cuda.empty_cache()
            gc.collect()

        # Post-processing
        if brightness != 1.0:
            temp = ImageEnhance.Brightness(result).enhance(brightness)
            result.close()
            result = temp
        if contrast != 1.0:
            result = ImageEnhance.Contrast(result).enhance(contrast)
        if saturation != 1.0:
            result = ImageEnhance.Color(result).enhance(saturation)
        if sharpness != 1.0:
            result = ImageEnhance.Sharpness(result).enhance(sharpness)
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
            result = result.point(table * 3)

        return result

def pad_to_multiple(tensor, multiple=16):
    """Pad H and W so they are divisible by `multiple` (required by deep architectures)."""
    import torch.nn.functional as F
    _, _, h, w = tensor.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        tensor = F.pad(tensor, (0, pw, 0, ph), mode="reflect")
    return tensor, h, w

# Keeping legacy functions for compatibility but wrapping class

def load_model(checkpoint_path, base_ch=32, num_dmrb=3, device="cpu"):
    em = EnhancementModel(checkpoint_path, device)
    em.loadModel(base_ch=base_ch)
    return em

def enhance_image(model_obj, pil_img, **kwargs):
    from utils.entities import LowLightImage
    # Mock preferences if passed as kwargs
    class Prefs: pass
    p = Prefs()
    for k, v in kwargs.items(): setattr(p, k, v)
    
    # Check if model_obj is actual model or EnhancementModel
    if isinstance(model_obj, EnhancementModel):
        return model_obj.enhanceImage(LowLightImage(pil_img, "PNG"), p)
    else:
        # Backward compat for direct model usage if needed
        # (Though we should update app.py)
        pass



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
    import torch
    from PIL import Image
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
