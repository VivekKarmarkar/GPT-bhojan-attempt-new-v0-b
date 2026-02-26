import base64
import os
import sys
import time

from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

from config import SAM_CHECKPOINT, OPENAI_API_KEY, DEVICE
from analysis import analyze_image
from vision import load_models, run_pipeline, image_to_base64

app = Flask(__name__)

# Global model references — loaded once at startup
yolo = None
sam_predictor = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    # Decode to PIL
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Base64 for GPT
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Step 1: GPT-4o analysis
    print("\n--- GPT-4o full-plate analysis ---")
    t0 = time.time()
    analysis = analyze_image(b64_image)
    print(f"  Analysis took {time.time() - t0:.1f}s")
    print(f"  Description: {analysis.get('description', '')[:80]}...")
    print(f"  Items: {analysis.get('items', [])}")

    # Step 2: Vision pipeline (YOLO + SAM + GPT classification/verification)
    print("\n--- Vision pipeline ---")
    t0 = time.time()
    vision_result = run_pipeline(pil_image, analysis, yolo, sam_predictor)
    print(f"  Pipeline took {time.time() - t0:.1f}s")

    return jsonify({
        "description": analysis.get("description", ""),
        "scores": {
            "health": analysis.get("health", 0),
            "satiety": analysis.get("satiety", 0),
            "bloat": analysis.get("bloat", 0),
            "tasty": analysis.get("tasty", 0),
            "addiction": analysis.get("addiction", 0),
        },
        "annotated_b64": vision_result["annotated_b64"],
        "items": vision_result["items"],
    })


if __name__ == "__main__":
    # Validate environment
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"ERROR: SAM checkpoint not found at {SAM_CHECKPOINT}")
        print("Run: bash setup.sh")
        sys.exit(1)

    print(f"Device: {DEVICE}")

    # Load models once
    yolo, sam_predictor = load_models()

    app.run(host="0.0.0.0", port=8080, debug=False)
