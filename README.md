# Low-Light Image Enhancement Tool
### Multi-Scale Progressive Fusion Network (MSPFN)
**Based on:** Zhang et al., *"Multi-Scale Progressive Fusion Network for Low-Light Image Enhancement"*, IEEE TIM 2025

---

## Project Structure

```
low_light_enhancement/
│
├── model/
│   ├── __init__.py
│   ├── network.py      ← Full MSPFN (Fig. 1)
│   ├── dmrb.py         ← DMRB block (Fig. 2, Eq. 1)
│   ├── fda.py          ← FDA block  (Fig. 4)
│   ├── sampling.py     ← Dual-channel superposition blocks
│   ├── attention.py    ← Coordinate Attention [37]
│   └── loss.py         ← Combined loss (Eq. 2–6)
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py      ← LOL-v1 dataset loader
│   └── metrics.py      ← PSNR / SSIM evaluation
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── templates/
│   └── index.html
│
├── train.py            ← Training script (paper exact settings)
├── inference.py        ← Single-image / batch inference
├── app.py              ← Flask web application
└── requirements.txt
```

---

## Architecture (exact paper)

### Network (Fig. 1)
1. 2 Conv layers → shallow features
2. Dual-channel downsampling × 2  →  3 scales (full / half / quarter)
3. **DMRB** at quarter resolution (lowest first)
4. Bottom-up: **UpSample** → **FDA** fusion → **DMRB** at half, then full
5. Splice all 3 scales at full resolution → 2 Conv → illumination estimate
6. `output = clamp(input + illumination, 0, 1)`

### DMRB (Eq. 1)
```
F_DMRB = DCA(F_IN)  +  W( IRB(F_CM) )
```
- **F_CM** : two grouped 3×3 convolutions on F_IN
- **IRB**  : parallel 3×3 + 5×5 group-conv, 1×1 fuse, two short connections (cyclic)
- **W**    : two 3×3 convolutions
- **DCA**  : two 3×3 convolutions + Coordinate Attention

### FDA (Fig. 4)
- Splice adjacent-scale features → Global AvgPool
- Conv branch (local perceptual) → Sigmoid weight w1
- Linear branch (global correlation) → Sigmoid weight w2
- `fused = x_low * w1 + x_high * w2` → 1×1 Conv

### Loss (Eq. 2–6)
```
L = 1·Lmse + 0.001·Lcolor + 0.1·Lssim + 1·Ltv
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset Setup (LOL-v1)

Download LOL-v1 and arrange as:
```
data/LOL/
  our485/
    low/    ← 485 low-light training images
    high/   ← 485 normal-light ground truth
  eval15/
    low/    ← 15 test images
    high/
```

---

## Training (exact paper settings)

```bash
python train.py \
  --data_root   data/LOL \
  --epochs      500      \
  --batch_size  32       \
  --lr          1e-4     \
  --base_ch     32       \
  --num_dmrb    3
```

Resume from checkpoint:
```bash
python train.py --data_root data/LOL --resume checkpoints/epoch_0050.pth
```

AMP (faster on modern GPUs):
```bash
python train.py --data_root data/LOL --mixed_prec
```

---

## Inference

```bash
# Single image
python inference.py \
  --input   dark_photo.jpg \
  --output  result.png \
  --checkpoint checkpoints/best.pth

# Folder
python inference.py \
  --input  images/low/ \
  --output images/enhanced/

# With user preference adjustments
python inference.py \
  --input dark.jpg --output out.png \
  --enhance 1.0       \   # 0=original, 1=full enhancement
  --brightness 1.2    \   # post-process brightness
  --contrast 1.1          # post-process contrast
```

---

## Web Application

```bash
python app.py --checkpoint checkpoints/best.pth
```
Open **http://localhost:5000** in your browser.

Features:
- Drag & drop or file browser upload (JPG / PNG / BMP)
- Enhancement strength slider (0–1)
- Brightness slider (0.5–3.0×)
- Contrast slider (0.5–3.0×)
- Side-by-side original vs enhanced comparison
- Download enhanced image

---

## Training Modes (LOL-v1 + LOL-v2)

`train.py` now supports three dataset modes:

- `lol_v1`: built-in `our485/eval15` split under `data_root`
- `custom`: explicit paired directories (recommended for LOL-v2 stage training)
- `mixed`: LOL-v1 train + second dataset train in one run

### Stage A: LOL-v2 pretraining (custom)

```bash
python train.py \
  --dataset custom \
  --data_root data/LOLv2 \
  --train_low_dir train/low \
  --train_high_dir train/high \
  --val_low_dir val/low \
  --val_high_dir val/high \
  --epochs 300 --lr 1e-4
```

### Stage B: LOL-v1 finetune

```bash
python train.py \
  --dataset lol_v1 \
  --data_root data/LOL \
  --resume checkpoints/best.pth \
  --epochs 500 --lr 5e-5
```

### Stage C: Mixed finetune (optional)

```bash
python train.py \
  --dataset mixed \
  --data_root data/LOL \
  --data_root_2 data/LOLv2 \
  --train_low_dir train/low \
  --train_high_dir train/high \
  --resume checkpoints/best.pth \
  --epochs 700 --lr 3e-5
```

---

## Paper Training Results on LOL-v1 (Table I)

| Method   | PSNR↑  | SSIM↑  |
|----------|--------|--------|
| LIME     | 15.97  | 0.577  |
| RetinexNet | 19.38 | 0.559 |
| DLN      | 21.39  | 0.810  |
| **Ours** | **22.57** | **0.826** |
