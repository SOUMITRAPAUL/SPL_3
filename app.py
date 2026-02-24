import sys
import os
import gc
import io
import traceback
from flask import Flask, render_template, request, jsonify

# Immediate startup signal for logs
print("--- FLASK STARTING (STABLE VERSION) ---", flush=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Shared model pointers
_model = None
_load_error = None

def get_mem_usage():
    """Returns VmRSS for Linux systems to track OOM risks."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if "VmRSS" in line:
                    return line.strip()
    except: return "unknown (non-linux?)"
    return "unknown"

def _ensure_model_loaded():
    global _model, _load_error
    if _model is not None:
        return True
    
    try:
        print(f"--- LAZY LOADING MODEL ({get_mem_usage()}) ---", flush=True)
        # Import inside to prevent startup timeout
        import inference
        
        # Determine checkpoint
        ckpt = "checkpoints/lolv2_test.pth" 
        for candidate in ["checkpoints/lolv2_test.pth", "checkpoints/best1.pth", "checkpoints/best.pth"]:
            if os.path.exists(candidate):
                ckpt = candidate
                break

        print(f"--- USING CHECKPOINT: {ckpt} ---", flush=True)
        _model = inference.load_model(ckpt, device="cpu")
        print(f"--- MODEL READY ({get_mem_usage()}) ---", flush=True)
        return True
    except Exception as e:
        _load_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"--- MODEL LOAD FAILED: {_load_error} ---", flush=True)
        return False

@app.route("/")
def index():
    print(f"--- SERVING INDEX ({get_mem_usage()}) ---", flush=True)
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "model_error": _load_error is not None,
        "mem": get_mem_usage(),
        "pid": os.getpid()
    })

@app.route("/debug")
def debug():
    results = {
        "cwd": os.getcwd(),
        "python": sys.version,
        "mem": get_mem_usage(),
        "files": os.listdir('.'),
        "checkpoints": os.listdir('checkpoints') if os.path.exists('checkpoints') else "MISSING",
        "load_error": _load_error
    }
    
    # Check imports
    try:
        import torch
        results["torch"] = torch.__version__
    except Exception as e: results["torch"] = str(e)

    try:
        import inference
        results["inference"] = "OK"
    except Exception as e: results["inference"] = str(e)

    return jsonify(results)

@app.route("/enhance", methods=["POST"])
def api_enhance():
    if not _ensure_model_loaded():
        return jsonify({"error": "Model failed to load", "details": _load_error}), 500
    
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        from PIL import Image
        import base64
        import inference

        img = Image.open(file.stream).convert("RGB")
        print(f"--- ENHANCING IMAGE: {img.size} ({get_mem_usage()}) ---", flush=True)
        
        # Max resolution safety for Free Tier
        MAX_DIM = 224
        if max(img.size) > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM))
            print(f"--- RESIZED TO: {img.size} ---", flush=True)

        strength = float(request.form.get("strength", 1.0))
        enhanced = inference.enhance_image(_model, img, enhance_strength=strength, device="cpu")
        
        # Convert to Base64
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        print(f"--- ENHANCE COMPLETE ({get_mem_usage()}) ---", flush=True)
        gc.collect() # Aggressive cleanup
        
        return jsonify({
            "image": f"data:image/png;base64,{b64_str}",
            "mem_after": get_mem_usage()
        })
    except Exception as e:
        print(f"--- ENHANCE ERROR: {str(e)} ---", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Local dev mode
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
