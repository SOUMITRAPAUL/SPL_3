import os
import io
import gc
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from PIL import Image

import inference

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

_model = None
_load_error = None

def _ensure_model_loaded():
    global _model, _load_error
    if _model is not None:
        return True
    
    try:
        ckpt = None
        for candidate in ["checkpoints/best1.pth", "checkpoints/best.pth", "checkpoints/lolv2_test.pth"]:
            if os.path.exists(candidate):
                ckpt = candidate
                break
        
        if not ckpt:
            raise FileNotFoundError("Could not find any .pth checkpoint in checkpoints/ folder.")

        print(f"--- Loading Model: {ckpt} ---", flush=True)
        _model = inference.load_model(ckpt, device="cpu")
        print("--- Model Loaded Successfully ---", flush=True)
        return True
    except Exception as e:
        _load_error = f"{type(e).__name__}: {str(e)}"
        print(f"--- FAILED TO LOAD MODEL: {_load_error} ---", flush=True)
        return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enhance", methods=["POST"])
def api_enhance():
    if not _ensure_model_loaded():
        return jsonify({"error": "Model load failed", "details": _load_error}), 500
    
    # Matching app.js: form.append('file', file);
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        w, h = img.size
        
        # Extract sliders matching app.js
        enhance_val = float(request.form.get("enhance", 1.0))
        brightness  = float(request.form.get("brightness", 1.0))
        contrast    = float(request.form.get("contrast", 1.0))
        sharpness   = float(request.form.get("sharpness", 1.0))
        gamma       = float(request.form.get("gamma", 1.0))
        saturation  = float(request.form.get("saturation", 1.0))
        denoise     = float(request.form.get("denoise", 0.0))
        auto_align  = request.form.get("auto_align") == 'true'

        print(f"--- Enhancing image: {file.name} ({w}x{h}) ---", flush=True)
        
        enhanced = inference.enhance_image(
            _model, img, 
            enhance_strength=enhance_val,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            gamma=gamma,
            saturation=saturation,
            denoise_weight=denoise,
            auto_align=auto_align,
            device="cpu"
        )
        
        # Convert to Base64
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        gc.collect()
        
        # Matching app.js expectations: data.enhanced_b64, data.width, data.height
        return jsonify({
            "enhanced_b64": b64_str,
            "width": w,
            "height": h
        })
    except Exception as e:
        print(f"--- ERROR: {str(e)} ---", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
