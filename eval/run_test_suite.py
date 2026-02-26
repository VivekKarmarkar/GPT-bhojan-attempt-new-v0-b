"""Automated test suite: upload all 17 sample images, collect results, save JSON."""
import json
import os
import time
import requests

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Sample Food Library")
URL = "http://localhost:8080/analyze"
OUTPUT = os.path.join(os.path.dirname(__file__), "test_results.json")

# Ground truth from ground_truth_labels.md
GROUND_TRUTH = {
    "sample_food_1": ["Palak Paneer", "Garlic/Plain Naan"],
    "sample_food_2": ["Roti/Chapati", "Rice/Biryani", "Chicken Curry", "Chicken Curry", "Lime", "Onion"],
    "sample_food_3": ["Sambar", "Chutney", "Wada", "Wada", "Wada"],
    "sample_food_4": ["Sambar", "Chutney", "Chutney", "Idli", "Idli"],
    "sample_food_5": ["Uttapam", "Sambar", "Chutney"],
    "sample_food_6": ["Subji", "Curd", "Dal", "Roti/Paratha", "Rice"],
    "sample_food_7": ["Rasmalai"],
    "sample_food_8": ["Gajar Halwa"],
    "sample_food_9": ["Pizza Slice", "Pizza Slice", "Pizza Slice"],
    "sample_food_10": ["Modak", "Modak", "Modak", "Modak", "Modak", "Modak", "Modak"],
    "sample_food_11": ["Banana Pudding"],
    "sample_food_12": ["Avocado Egg Toast"],
    "sample_food_13": ["Soup", "Sandwich"],
    "sample_food_14": ["Salmon", "Salmon", "Salmon", "Salmon"],
    "sample_food_15": ["Gajar Halwa"],
    "sample_food_16": ["Rasmalai"],
    "sample_food_17": ["Gajar Halwa"],
}

# Map filenames (without extension) to actual files
def find_images():
    files = os.listdir(SAMPLE_DIR)
    mapping = {}
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = os.path.splitext(f)[0]
            mapping[stem] = os.path.join(SAMPLE_DIR, f)
    return mapping

def main():
    images = find_images()
    results = []

    for i in range(1, 18):
        key = f"sample_food_{i}"
        path = images.get(key)
        if not path:
            print(f"SKIP {key}: file not found")
            continue

        gt = GROUND_TRUTH.get(key, [])
        print(f"\n{'='*60}")
        print(f"Testing {key} (GT: {gt})")
        print(f"{'='*60}")

        t0 = time.time()
        with open(path, "rb") as f:
            resp = requests.post(URL, files={"image": (os.path.basename(path), f)})
        elapsed = time.time() - t0

        if resp.status_code != 200:
            print(f"  ERROR: status {resp.status_code}")
            results.append({
                "image": key,
                "ground_truth": gt,
                "error": resp.status_code,
                "elapsed_s": round(elapsed, 1),
            })
            continue

        data = resp.json()
        detected_items = [item["name"] for item in data.get("items", [])]
        scores = data.get("scores", {})
        description = data.get("description", "")

        print(f"  Description: {description[:80]}...")
        print(f"  Detected items: {detected_items}")
        print(f"  GT items:       {gt}")
        print(f"  Scores: {scores}")
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            "image": key,
            "ground_truth": gt,
            "gt_count": len(gt),
            "detected_items": detected_items,
            "detected_count": len(detected_items),
            "description": description,
            "scores": scores,
            "elapsed_s": round(elapsed, 1),
        })

    # Save results
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {OUTPUT}")
    print(f"Total images tested: {len(results)}")

if __name__ == "__main__":
    main()
