print("--- APP STARTUP ---")
"""
Flask web application — Low-Light Image Enhancement Tool.
Heavy imports (torch, model, etc.) are deferred until the first /enhance
request so gunicorn can bind the port without OOM-crashing immediately.
"""

import base64
import gc
import io
import os
import threading

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

# ── Lazy-loaded globals ────────────────────────────────────────────────────────
_model      = None
_device     = None
_enhance_fn = None
_model_lock = threading.Lock()
_load_error = None


def _ensure_model_loaded():
    """Load heavy deps + model on the very first request. Thread-safe."""
    global _model, _device, _enhance_fn, _load_error

    if _model is not None or _load_error is not None:
        return

    with _model_lock:
        if _model is not None or _load_error is not None:
            return
        try:
            import torch
            from inference import enhance_image, load_model

            _device     = torch.device("cpu")   # always CPU on free tier
            _enhance_fn = enhance_image

            paths = [
                "checkpoints/best1.pth",
                "checkpoints/best.pth",
                "best1.pth",
                "best.pth",
            ]
            ckpt = next((p for p in paths if os.path.exists(p)), None)
            if ckpt is None:
                raise FileNotFoundError("No checkpoint found: " + str(paths))

            print(f"[lazy] Loading model from {ckpt} ...")
            _model = load_model(ckpt, 32, 3, _device)
            gc.collect()   # free any temp objects from loading
            print("[lazy] Model READY")

        except BaseException as exc:   # catches MemoryError too
            err_msg = f"{type(exc).__name__}: {exc}"
            _load_error = err_msg
            print(f"[CRITICAL] Model load failed: {err_msg}")
            raise RuntimeError(err_msg) from None   # convert to Exception


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── Global error handler (catches everything including MemoryError) ────────────
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def api_enhance():
    # ── Load model (first call only) ──
    try:
        _ensure_model_loaded()
    except Exception as e:
        return jsonify({"error": f"Model load failed: {str(e)}"}), 503

    if _load_error:
        return jsonify({"error": _load_error}), 503

    # ── Validate upload ──
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # ── Enhance ──
    try:
        from PIL import Image

        enhance  = max(0.0, min(float(request.form.get("enhance",    1.0)), 1.0))
        bright   = max(0.5, min(float(request.form.get("brightness", 1.5)), 3.0))
        contrast = max(0.5, min(float(request.form.get("contrast",   1.0)), 3.0))
        gamma    = max(0.5, min(float(request.form.get("gamma",      1.3)), 2.5))
        sat      = max(0.0, min(float(request.form.get("saturation", 1.0)), 2.5))
        sharp    = max(0.2, min(float(request.form.get("sharpness",  2.0)), 3.0))
        denoise  = max(0.0, min(float(request.form.get("denoise",    0.0)), 0.2))
        auto_align = request.form.get("auto_align", "false").lower() == "true"

        pil_img = Image.open(f.stream).convert("RGB")
        w, h    = pil_img.size

        # Keep well under 512 MB RAM limit — 256 px on longest side
        MAX_DIM = 256
        if max(w, h) > MAX_DIM:
            scale   = MAX_DIM / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        gc.collect()   # free memory before inference

        result = _enhance_fn(
            _model, pil_img,
            enhance_strength=enhance,
            brightness=bright,
            contrast=contrast,
            gamma=gamma,
            saturation=sat,
            sharpness=sharp,
            denoise_weight=denoise,
            auto_align=auto_align,
            device=_device,
        )

        gc.collect()   # free activation tensors after inference

        out_w, out_h = result.size
        return jsonify({
            "enhanced_b64": pil_to_b64(result),
            "width": out_w,
            "height": out_h,
        })

    except MemoryError:
        gc.collect()
        return jsonify({"error": "Out of memory — try a smaller image"}), 503

    except BaseException as e:   # catch everything
        import traceback
        traceback.print_exc()
        gc.collect()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": _model is not None,
        "load_error":   _load_error,
    })


@app.route("/debug")
def debug():
    """Test each import individually to find which one fails."""
    results = {}
    libs = [
        ("torch",          "import torch; results['torch'] = torch.__version__"),
        ("einops",         "import einops"),
        ("pywt",           "import pywt"),
        ("cv2",            "import cv2"),
        ("PIL",            "from PIL import Image"),
        ("skimage",        "from skimage.restoration import denoise_tv_chambolle"),
        ("timm",           "import timm; results['timm'] = timm.__version__"),
        ("timm.layers",    "from timm.layers import trunc_normal_"),
        ("model",          "from model import DRSformer"),
        ("inference",      "from inference import load_model"),
    ]
    for name, code in libs:
        try:
            exec(code, {"results": results})
            results[name] = results.get(name, "OK")
        except BaseException as e:
            results[name] = f"FAILED: {type(e).__name__}: {e}"
    return jsonify(results)



# ── Local dev entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    _ensure_model_loaded()
    app.run(host=args.host, port=args.port, debug=args.debug)
