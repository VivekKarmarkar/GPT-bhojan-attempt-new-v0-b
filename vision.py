import base64
import io
import random

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

from config import SAM_CHECKPOINT, SAM_MODEL_TYPE, YOLO_MODEL, DEVICE
from analysis import classify_and_validate_crop, verify_mask


def load_models():
    """Load YOLO and SAM once at startup. Returns (yolo, sam_predictor)."""
    print(f"Loading YOLO ({YOLO_MODEL}) on {DEVICE}...")
    yolo = YOLO(YOLO_MODEL)

    print(f"Loading SAM ({SAM_MODEL_TYPE}) on {DEVICE}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)

    print("Models loaded.")
    return yolo, predictor


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_cv2(img_arr: np.ndarray) -> str:
    """Convert BGR numpy array to base64 PNG string."""
    rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_color_map = {}

def _get_color(label: str) -> tuple:
    """Consistent random bright color per label."""
    if label not in _color_map:
        _color_map[label] = tuple(random.randint(100, 255) for _ in range(3))
    return _color_map[label]


def _compute_iou(box_a, box_b):
    """Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _dedup_boxes(classified, iou_threshold=0.5):
    """Remove duplicate boxes: same label + high IoU → keep smaller box.

    classified: list of (idx, x1, y1, x2, y2, label, area)
    Returns filtered list with duplicates removed.
    """
    # Group by normalized label
    from collections import defaultdict
    by_label = defaultdict(list)
    for entry in classified:
        by_label[entry[5].lower()].append(entry)

    survivors = []
    for label, group in by_label.items():
        # Sort by area ascending — prefer tighter boxes
        group.sort(key=lambda e: e[6])
        kept = []
        for entry in group:
            box = entry[1:5]
            # Check if this box overlaps heavily with any already-kept box
            duplicate = False
            for kept_entry in kept:
                kept_box = kept_entry[1:5]
                if _compute_iou(box, kept_box) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(entry)
            else:
                print(f"  Box {entry[0]}: DEDUPED — overlaps with kept box for '{entry[5]}'", flush=True)
        survivors.extend(kept)

    return survivors


def run_pipeline(pil_image: Image.Image, analysis: dict, yolo, predictor) -> dict:
    """Full vision pipeline: YOLO detect -> GPT classify -> dedup -> SAM segment -> GPT verify -> overlay.

    Returns {annotated_b64, items: [{name, color}]}.
    """
    img_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_orig = img_cv2.copy()
    img_orig_c = img_cv2.copy()
    img_h, img_w = img_cv2.shape[:2]
    img_area = img_h * img_w

    description = analysis.get("description", "")

    # YOLO detection
    print("Running YOLO detection...")
    results = yolo(img_cv2)
    boxes = results[0].boxes.xyxy
    print(f"  Found {len(boxes)} bounding boxes")

    # Base64 of full image (computed once, reused per box)
    full_b64 = image_to_base64(pil_image)

    # --- Pass 1: GPT classification for every box ---
    print("Pass 1: GPT classification...")
    classified = []  # (idx, x1, y1, x2, y2, label, box_area)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / img_area

        # Cost guard: skip boxes covering >90% of the image
        if area_ratio > 0.9:
            print(f"  Box {i}: SKIPPED — box covers {area_ratio:.0%} of image (cost guard)", flush=True)
            continue

        crop_pil = pil_image.crop((x1, y1, x2, y2))
        crop_b64 = image_to_base64(crop_pil)
        label = classify_and_validate_crop(full_b64, crop_b64, area_ratio, description)
        print(f"  Box {i}: {area_ratio:.0%} of image → classified as '{label}'", flush=True)

        if label.lower() == "none":
            continue

        classified.append((i, x1, y1, x2, y2, label, box_area))

    # --- Dedup: same label + high IoU → keep tighter box ---
    print(f"  {len(classified)} boxes classified — deduplicating...")
    survivors = _dedup_boxes(classified)
    print(f"  {len(survivors)} boxes after dedup")

    # --- Pass 2: SAM + A/B verify + overlay for survivors only ---
    print("Pass 2: SAM segmentation + verification...")
    predictor.set_image(np.array(pil_image))

    items = []

    for idx, x1, y1, x2, y2, label, _ in survivors:
        color = _get_color(label)

        # SAM segmentation with box prompt
        input_box = np.array([[x1, y1, x2, y2]])
        masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        mask_c = np.logical_not(mask)

        # Extract mask and complement crops
        segmented = np.zeros_like(img_orig)
        segmented[mask] = img_orig[mask]
        segmented_c = np.zeros_like(img_orig_c)
        segmented_c[mask_c] = img_orig_c[mask_c]

        crop_a = segmented[y1:y2, x1:x2]
        crop_b = segmented_c[y1:y2, x1:x2]

        # Brightness check
        gray_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2GRAY)
        bright_a = np.sum(gray_a > 30) > 100
        bright_b = np.sum(gray_b > 30) > 100

        # GPT A/B verification
        img_a_b64 = _encode_cv2(crop_a)
        img_b_b64 = _encode_cv2(crop_b)
        choice = verify_mask(label, img_a_b64, img_b_b64)
        print(f"  Box {idx}: GPT chose '{choice}'", flush=True)

        mask_final = None
        if choice == "A" and bright_a:
            mask_final = mask
        elif choice == "B" and bright_b:
            # Invert only within the bounding box — not the entire image
            bbox_mask = np.zeros_like(mask)
            bbox_mask[y1:y2, x1:x2] = True
            mask_final = np.logical_and(mask_c, bbox_mask)

        # Overlay mask with alpha blend
        if mask_final is not None:
            overlay_color = np.array(color, dtype=np.uint8)
            img_cv2[mask_final] = (
                0.6 * overlay_color + 0.4 * img_cv2[mask_final]
            ).astype(np.uint8)

        # Draw bounding box and label
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=color, thickness=3)
        cv2.putText(img_cv2, label, (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        items.append({"name": label, "color": f"rgb({color[0]},{color[1]},{color[2]})"})

    # Encode annotated image
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(img_rgb)
    annotated_b64 = image_to_base64(annotated_pil)

    print(f"Pipeline complete: {len(items)} items segmented")
    return {"annotated_b64": annotated_b64, "items": items}
