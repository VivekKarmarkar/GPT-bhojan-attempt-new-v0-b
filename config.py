import os
import torch
from dotenv import load_dotenv

load_dotenv()


def get_device():
    """Auto-detect CUDA with a real allocation test, fallback to CPU."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            pass
    return "cpu"


DEVICE = get_device()
SAM_CHECKPOINT = os.path.join("models", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
YOLO_MODEL = "yolov8m.pt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
