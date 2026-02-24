import sys
import os
import base64
import gc
import io
import threading
import traceback
from flask import Flask, render_template, request, jsonify

# Immediate startup signal for logs
print("--- FLASK BOOTSTRAP STARTING ---", flush=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── Globals ──────────────────────────────────────────────────────────────────
_model      = None
_device     = None
_enhance_fn = None
_model_lock = threading.Lock()
_load_error = None

# sys.exit interceptor
def _intercept_exit(code=0):
    print(f"\n--- [INTERCEPTOR] SYSTEM EXIT CALLED WITH CODE: {code} ---", flush=True)
    traceback.print_stack()
    raise RuntimeError(f"Intercepted sys.exit({code})")

def _ensure_model_loaded():
    """Deferred loading with breadcrumbs to find the 'SystemExit' trigger."""
    global _model, _device, _enhance_fn, _load_error

    if _model is not None or _load_error is not None:
        return

    with _model_lock:
        if _model is not None or _load_error is not None:
            return
        
        old_exit = sys.exit
        sys.exit = _intercept_exit
        
        try:
            print("[lazy] STEP 1: Importing Torch", flush=True)
            import torch
            _device = torch.device("cpu")
            
            print("[lazy] STEP 2: Importing Inference Utils", flush=True)
            from inference import enhance_image, load_model
            _enhance_fn = enhance_image

            print("[lazy] STEP 3: Identifying Checkpoint", flush=True)
            paths = ["checkpoints/best1.pth", "checkpoints/best.pth", "best1.pth", "best.pth"]
            ckpt_path = next((p for p in paths if os.path.exists(p)), None)
            
            if not ckpt_path:
                print("[lazy] ERROR: No checkpoint file found in workspace", flush=True)
                raise FileNotFoundError(f"Checkpoints missing! Checked {paths}")

            print(f"[lazy] STEP 4: Calling load_model({ckpt_path})", flush=True)
            # This is the most likely place for SystemExit if sub-libraries fail
            _model = load_model(ckpt_path, 32, 3, _device)
            
            print("[lazy] STEP 5: Success!", flush=True)
            gc.collect()

        except BaseException as e:
            err = f"{type(e).__name__}: {str(e)}"
            print(f"[CRITICAL] lazy loading failed at {type(e).__name__}: {str(e)}", flush=True)
            traceback.print_exc()
            _load_error = err
        finally:
            sys.exit = old_exit

# ── Error Handlers ────────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"--- APP EXCEPTION: {e} ---", flush=True)
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
    
    old_exit = sys.exit
    sys.exit = _intercept_exit
    
    # Check basic imports
    tests = [
        ("torch",      "import torch"),
        ("inference",  "import inference"),
        ("model",      "from model import DRSformer"),
        ("checkpoint", "import os; results['checkpoint_exists'] = any(os.path.exists(p) for p in ['checkpoints/best1.pth', 'best1.pth', 'best.pth'])")
    ]
    
    for name, cmd in tests:
        try:
            local_scope = {"results": results}
            exec(cmd, {}, local_scope)
            if name not in results: results[name] = "OK"
        except BaseException as e:
            results[name] = f"ERROR: {type(e).__name__}: {str(e)}"
            
    sys.exit = old_exit
    return jsonify(results)

@app.route("/enhance", methods=["POST"])
def api_enhance():
    try:
        _ensure_model_loaded()
    except BaseException as e:
        return jsonify({"error": f"Load crash: {type(e).__name__}: {str(e)}"}), 503

    if _load_error:
        return jsonify({"error": f"Model failed to start: {_load_error}"}), 503

    file = request.files.get("file")
    if not file: return jsonify({"error": "No file"}), 400
    
    try:
        from PIL import Image
        img = Image.open(file.stream).convert("RGB")
        
        # Scaling to fit RAM
        LIMIT = 224
        if max(img.size) > LIMIT:
            scale = LIMIT / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
        
        gc.collect()
        res = _enhance_fn(_model, img, 1.0, 1.5, 1.0, 1.3, 1.0, 2.0, 0.0, False, _device)
        
        buf = io.BytesIO()
        res.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        return jsonify({"enhanced_b64": b64, "width": res.width, "height": res.height})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
