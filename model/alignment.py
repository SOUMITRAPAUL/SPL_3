import numpy as np
import cv2
from PIL import Image

def gray_world_wb(img_np):
    """
    Simple Gray-World White Balance.
    Optimized for memory efficiency.
    """
    import gc
    # Calculate means on uint8 to save memory
    avg_r = np.mean(img_np[:, :, 0])
    avg_g = np.mean(img_np[:, :, 1])
    avg_b = np.mean(img_np[:, :, 2])
    
    avg_gray = (avg_r + avg_g + avg_b) / 3.0
    
    # Scale factors
    s_r = avg_gray / (avg_r + 1e-8)
    s_g = avg_gray / (avg_g + 1e-8)
    s_b = avg_gray / (avg_b + 1e-8)
    
    # Create float array only once
    img_float = img_np.astype(np.float32)
    img_float[:, :, 0] *= s_r
    img_float[:, :, 1] *= s_g
    img_float[:, :, 2] *= s_b
    
    out = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # Explicit cleanup
    del img_float
    gc.collect()
    
    return out

def align_to_lol_domain(img_np):
    """
    Histogram alignment: Compresses image intensity to match the typical 
    low-light profile the model was trained on.
    """
    import gc
    # Convert to LAB to only touch Lightness
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Standard LOL-v1 images are often very compressed in the [0, 50] range
    mean_l = np.mean(l)
    if mean_l > 40:
        # Scale L channel down so model sees a "low light" image
        l_float = l.astype(np.float32)
        l = (l_float * (30.0 / mean_l)).astype(np.uint8)
        del l_float
        
    aligned = cv2.merge((l, a, b))
    out = cv2.cvtColor(aligned, cv2.COLOR_LAB2RGB)
    
    # Cleanup
    del lab, l, a, b, aligned
    gc.collect()
    
    return out

def apply_alignment(pil_img):
    """
    Main entry point for Domain Alignment.
    """
    img_np = np.array(pil_img)
    
    # 1. Fix color tints (camera sensor diffs)
    img_np = gray_world_wb(img_np)
    
    # 2. Align exposure (camera exposure diffs)
    img_np = align_to_lol_domain(img_np)
    
    return Image.fromarray(img_np)
