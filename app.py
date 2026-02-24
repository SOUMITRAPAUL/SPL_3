import os
import io
import gc
import base64
import traceback
from flask import Flask, render_template, request, jsonify
from PIL import Image

# For local development, we can import things more directly if we want,
# but keeping the lazy loading is still good for app responsiveness!
import inference

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32MB local limit

_model = None
_load_error = None

def _ensure_model_loaded():
    global _model, _load_error
    if _model is not None:
        return True
    
    try:
        # Check for multiple possible checkpoint names
        ckpt = None
        for candidate in ["checkpoints/best1.pth", "checkpoints/best.pth", "checkpoints/lolv2_test.pth"]:
            if os.path.exists(candidate):
                ckpt = candidate
                break
        
        if not ckpt:
            raise FileNotFoundError("Could not find any .pth checkpoint in checkpoints/ folder.")

        print(f"--- Loading Model: {ckpt} ---")
        _model = inference.load_model(ckpt, device="cpu")
        print("--- Model Loaded Successfully ---")
        return True
    except Exception as e:
        _load_error = f"{type(e).__name__}: {str(e)}"
        print(f"--- FAILED TO LOAD MODEL: {_load_error} ---")
        return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enhance", methods=["POST"])
def api_enhance():
    if not _ensure_model_loaded():
        return jsonify({"error": "Model load failed", "details": _load_error}), 500
    
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        strength = float(request.form.get("strength", 1.0))
        
        print(f"--- Enhancing image ({img.size}) ---")
        
        # We removed the 224px limit for local machines!
        enhanced = inference.enhance_image(_model, img, enhance_strength=strength, device="cpu")
        
        # Save to buffer
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        gc.collect()
        return jsonify({"image": f"data:image/png;base64,{b64_str}"})
    except Exception as e:
        print(f"--- ERROR: {str(e)} ---")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start the app locally
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
