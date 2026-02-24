"""
Flask web application — Low-Light Image Enhancement Tool.

Run (after training):
  python app.py --checkpoint checkpoints/best.pth

Exposes:
  GET  /           → main UI page
  POST /enhance    → JSON API: upload image + sliders → base64 result
  GET  /health     → model status
"""

import argparse
import base64
import io
import os
import threading

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image

from model    import DRSformer
from inference import enhance_image, load_model, pad_to_multiple

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB

_model      = None
_device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_lock = threading.Lock()


def pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def api_enhance():
    _ensure_model_loaded()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        return jsonify({"error": f"Unsupported format '{ext}'. Use JPG, PNG, WEBP or BMP."}), 400

    try:
        enhance  = float(request.form.get("enhance",    1.0))
        bright   = float(request.form.get('brightness', 1.5))
        contrast = float(request.form.get('contrast', 1.0))
        gamma    = float(request.form.get('gamma', 1.3))
        sat      = float(request.form.get('saturation', 1.0))
        sharp    = float(request.form.get('sharpness', 2.0))
        denoise  = float(request.form.get('denoise', 0.00))
        auto_align = request.form.get("auto_align", "false").lower() == "true"

        if _model is None:
            return jsonify({"error": "Model not loaded. Train first, then restart app."}), 503

        # Clamp values
        enhance  = max(0.0, min(enhance, 1.0))
        bright   = max(0.5, min(bright, 3.0))
        contrast = max(0.5, min(contrast, 3.0))
        gamma    = max(0.5, min(gamma, 2.5))
        sat      = max(0.0, min(sat, 2.5))
        sharp    = max(0.2, min(sharp, 3.0))
        denoise  = max(0.0, min(denoise, 0.2))

        pil_img = Image.open(f.stream).convert("RGB")
        w, h = pil_img.size
        
        # QUALITY FIX: Maintain aspect ratio
        MAX_DIM = 600
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        result = enhance_image(_model, pil_img,
                                enhance_strength=enhance,
                                brightness=bright,
                                contrast=contrast,
                                gamma=gamma,
                                saturation=sat,
                                sharpness=sharp,
                                denoise_weight=denoise,
                                auto_align=auto_align,
                                device=_device)

        out_w, out_h = result.size
        return jsonify({
            "enhanced_b64": pil_to_b64(result),
            "width": out_w, "height": out_h,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    _ensure_model_loaded()
    return jsonify({"status": "ok", "model_loaded": _model is not None})


# ── Lazy model loader (called on first request) ───────────────────────────────
def _ensure_model_loaded():
    """Load the model on the first request. Thread-safe."""
    global _model
    if _model is not None:
        return
    with _model_lock:
        if _model is not None:   # double-checked locking
            return
        paths = [
            "checkpoints/best1.pth",
            "checkpoints/best.pth",
            "best1.pth",
            "best.pth",
        ]
        ckpt_path = next((p for p in paths if os.path.exists(p)), None)
        if ckpt_path:
            print(f"[lazy] Loading model from {ckpt_path} ...")
            _model = load_model(ckpt_path, 32, 3, _device)
            print("[lazy] Model READY")
        else:
            print("[CRITICAL] No checkpoint found. Enhancement will fail.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host",  type=str, default="0.0.0.0")
    p.add_argument("--port",  type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    _ensure_model_loaded()   # load eagerly for local dev
    app.run(host=args.host, port=args.port, debug=args.debug)
