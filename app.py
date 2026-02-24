import sys
import os
import gc
import io
import traceback
from flask import Flask, render_template, request, jsonify

# Immediate startup signal for logs
print("--- STARTING PRODUCTION APP ---", flush=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

_model = None
_load_error = None

def _ensure_model_loaded():
    global _model, _load_error
    if _model is not None:
        return True
    
    try:
        import inference
        ckpt = "checkpoints/best1.pth"
        if not os.path.exists(ckpt):
            ckpt = "checkpoints/lolv2_test.pth"
        
        print(f"--- LOADING MODEL: {ckpt} ---", flush=True)
        _model = inference.load_model(ckpt, device="cpu")
        print("--- MODEL READY ---", flush=True)
        return True
    except Exception as e:
        _load_error = f"{type(e).__name__}: {str(e)}"
        print(f"--- LOAD FAILED: {_load_error} ---", flush=True)
        return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "ready": _model is not None})

@app.route("/debug")
def debug():
    try:
        import torch
        torch_v = torch.__version__
    except: torch_v = "error"
    
    return jsonify({
        "python": sys.version,
        "torch": torch_v,
        "model_error": _load_error,
        "checkpoints": os.listdir('checkpoints') if os.path.exists('checkpoints') else "none"
    })

@app.route("/enhance", methods=["POST"])
def api_enhance():
    if not _ensure_model_loaded():
        return jsonify({"error": "Model load failed", "details": _load_error}), 500
    
    file = request.files.get("image")
    if not file: return jsonify({"error": "No image"}), 400

    try:
        from PIL import Image
        import base64
        import inference

        img = Image.open(file.stream).convert("RGB")
        
        # Free Tier Safety
        if max(img.size) > 224:
            img.thumbnail((224, 224))

        strength = float(request.form.get("strength", 1.0))
        enhanced = inference.enhance_image(_model, img, enhance_strength=strength, device="cpu")
        
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        gc.collect()
        return jsonify({"image": f"data:image/png;base64,{b64_str}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
