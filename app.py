import os
import io
import gc
import base64
import traceback
import threading
import uuid
from flask import Flask, render_template, request, jsonify
from PIL import Image
import inference
from dataclasses import dataclass
from utils.entities import LowLightImage, EnhancedImage

@dataclass
class UserPreferences:
    enhance_strength: float = 1.0
    brightness: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    denoise_weight: float = 0.0
    auto_align: bool = False

    def updatePreferences(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

class WebApplication:
    def __init__(self):
        self.supportedFormats = ['JPG', 'PNG', 'WEBP', 'BMP', 'TIFF']
        self.uiState = "idle"
        self.model_wrapper = None
        self.load_error = None

    def _ensure_model_loaded(self):
        if self.model_wrapper is not None:
            return True
        
        try:
            ckpt = None
            for candidate in ["checkpoints/best1.pth", "checkpoints/best.pth", "checkpoints/lolv2_test.pth", "best.pth"]:
                if os.path.exists(candidate):
                    ckpt = candidate
                    break
            
            if not ckpt:
                raise FileNotFoundError("Could not find any .pth checkpoint in project root or checkpoints/ folder.")

            from inference import EnhancementModel
            self.model_wrapper = EnhancementModel(ckpt, device="cpu")
            self.model_wrapper.loadModel()
            return True
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {str(e)}"
            return False

    def startEnhancement(self, file_storage, prefs_dict):
        import gc
        gc.collect() 
        
        if not self._ensure_model_loaded():
            raise RuntimeError(f"Model load failed: {self.load_error}")
        
        img = Image.open(file_storage.stream).convert("RGB")
        img = img.resize((600, 400), Image.LANCZOS)
        
        low_light_image = LowLightImage(data=img, format=file_storage.content_type)
        user_prefs = UserPreferences(**prefs_dict)
        
        try:
            print(f"--- Starting Enhancement (Strength: {user_prefs.enhance_strength}) ---")
            enhanced_pil = self.model_wrapper.enhanceImage(low_light_image, user_prefs)
            
            print("Encoding result to Base64...")
            buf = io.BytesIO()
            enhanced_pil.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            # Clean up immediately after encoding
            if hasattr(enhanced_pil, 'close'): enhanced_pil.close()
            buf.close()
            
            gc.collect()
            return {
                "enhanced_b64": b64_str,
                "width": 600,
                "height": 400
            }
        except Exception as e:
            print(f"!!! Error in startEnhancement: {str(e)}")
            raise e
        finally:
            img.close()
            del img
            gc.collect()
            print("--- Enhancement Step Finished ---")

web_app = WebApplication()
processing_tasks = {}

def background_enhance(task_id, file_bytes, content_type, prefs_dict):
    print(f"=== Background Task Started: {task_id} ===")
    try:
        class DummyFile:
            def __init__(self, b, c):
                self.stream = io.BytesIO(b)
                self.content_type = c
            def close(self): self.stream.close()
        
        dummy = DummyFile(file_bytes, content_type)
        result = web_app.startEnhancement(dummy, prefs_dict)
        processing_tasks[task_id] = {"status": "complete", "result": result}
        dummy.close()
        print(f"=== Background Task Success: {task_id} ===")
    except Exception as e:
        print(f"=== Background Task FAILED: {task_id} ===")
        traceback.print_exc()
        processing_tasks[task_id] = {"status": "error", "message": str(e)}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enhance", methods=["POST"])
def api_enhance():
    file = request.files.get("file")
    if not file: return jsonify({"error": "No image uploaded"}), 400

    try:
        task_id = str(uuid.uuid4())
        file_bytes = file.read()
        content_type = file.content_type
        
        prefs_dict = {
            "enhance_strength": float(request.form.get("enhance", 1.0)),
            "brightness": float(request.form.get("brightness", 1.0)),
            "contrast": float(request.form.get("contrast", 1.0)),
            "sharpness": float(request.form.get("sharpness", 1.0)),
            "gamma": float(request.form.get("gamma", 1.0)),
            "saturation": float(request.form.get("saturation", 1.0)),
            "denoise_weight": float(request.form.get("denoise", 0.0)),
            "auto_align": request.form.get("auto_align") == 'true'
        }

        processing_tasks[task_id] = {"status": "processing"}
        threading.Thread(target=background_enhance, args=(task_id, file_bytes, content_type, prefs_dict)).start()
        return jsonify({"task_id": task_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/task_status/<task_id>")
def get_task_status(task_id):
    task = processing_tasks.get(task_id)
    if not task: return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 50000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
