import sys
import os
import base64
import gc
import io
import threading
import traceback
from flask import Flask, render_template, request, jsonify

# Immediate startup signal
print("--- FLASK BOOTSTRAP STARTING ---", flush=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── Globals ──────────────────────────────────────────────────────────────────
_model      = None
_device     = None
_enhance_fn = None
_model_lock = threading.Lock()
_load_error = None

def _ensure_model_loaded():
    """Deferred loading of heavy AI dependencies."""
    global _model, _device, _enhance_fn, _load_error

    if _model is not None or _load_error is not None:
        return

    with _model_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            print("[lazy] Importing heavy libs (torch, etc)...", flush=True)
            import torch
            from inference import enhance_image, load_model

            _device     = torch.device("cpu")
            _enhance_fn = enhance_image

            paths = ["checkpoints/best1.pth", "checkpoints/best.pth", "best1.pth", "best.pth"]
            ckpt_path = next((p for p in paths if os.path.exists(p)), None)
            
            if not ckpt_path:
                raise FileNotFoundError(f"Checkpoints not found in {paths}")

            print(f"[lazy] Loading weights from {ckpt_path}...", flush=True)
            _model = load_model(ckpt_path, 32, 3, _device)
            gc.collect()
            print("[lazy] Model successfully loaded!", flush=True)

        except BaseException as e:
            err = f"{type(e).__name__}: {str(e)}"
            print(f"[CRITICAL] Model load failure: {err}", flush=True)
            traceback.print_exc()
            _load_error = err
            raise

# ── Error Handlers ────────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"--- SERVER ERROR: {e} ---", flush=True)
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    print("--- SERVING INDEX ---", flush=True)
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": _model is not None,
        "load_error": _load_error
    })

@app.route("/debug")
def debug():
    print("--- SERVING DEBUG ---", flush=True)
    results = {}
    test_libs = ["torch", "einops", "pywt", "cv2", "PIL", "skimage", "timm", "model", "inference"]
    for lib in test_libs:
        try:
            if lib == "PIL": exec("from PIL import Image")
            elif lib == "skimage": exec("from skimage.restoration import denoise_tv_chambolle")
            else: exec(f"import {lib}")
            results[lib] = "OK"
        except BaseException as e:
            results[lib] = f"ERROR: {type(e).__name__}: {e}"
    return jsonify(results)

@app.route("/enhance", methods=["POST"])
def api_enhance():
    print("--- ENHANCE REQUEST RECEIVED ---", flush=True)
    try:
        _ensure_model_loaded()
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    if _load_error:
        return jsonify({"error": _load_error}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        from PIL import Image
        
        # Read parameters
        strength = float(request.form.get("enhance", 1.0))
        bright   = float(request.form.get("brightness", 1.5))
        contrast = float(request.form.get("contrast", 1.0))
        gamma    = float(request.form.get("gamma", 1.3))
        sat      = float(request.form.get("saturation", 1.0))
        sharp    = float(request.form.get("sharpness", 2.0))
        denoise  = float(request.form.get("denoise", 0.0))
        align    = request.form.get("auto_align", "false").lower() == "true"

        img = Image.open(file.stream).convert("RGB")
        w, h = img.size
        
        # Power-save mode for Render Free Tier (256px limit)
        LIMIT = 256
        if max(w, h) > LIMIT:
            scale = LIMIT / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        
        gc.collect()
        
        enhanced = _enhance_fn(
            _model, img,
            enhance_strength=strength,
            brightness=bright,
            contrast=contrast,
            gamma=gamma,
            saturation=sat,
            sharpness=sharp,
            denoise_weight=denoise,
            auto_align=align,
            device=_device
        )
        
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        gc.collect()
        
        return jsonify({
            "enhanced_b64": b64,
            "width": enhanced.width,
            "height": enhanced.height
        })

    except Exception as e:
        print(f"--- ENHANCE ERROR: {e} ---", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Local only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
