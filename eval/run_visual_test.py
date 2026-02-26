"""Full visual test suite: upload all 17 images, save annotated results + composite cards."""
import json
import os
import base64
import time
import requests
import io
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Sample Food Library")
URL = "http://localhost:8080/analyze"
VIS_DIR = os.path.join(os.path.dirname(__file__), "visual_results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "test_figures")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "test_results.json")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

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


def find_images():
    files = os.listdir(SAMPLE_DIR)
    mapping = {}
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = os.path.splitext(f)[0]
            mapping[stem] = os.path.join(SAMPLE_DIR, f)
    return mapping


def fuzzy_match(detected, gt):
    d = detected.lower().strip()
    g = gt.lower().strip()
    if d in g or g in d:
        return True
    d_words = set(d.replace("/", " ").replace("-", " ").split())
    g_words = set(g.replace("/", " ").replace("-", " ").split())
    meaningful = d_words.intersection(g_words) - {"ka", "a", "the", "or", "and", "with"}
    return len(meaningful) > 0


def create_result_card(original_path, annotated_b64, description, items, scores, gt, image_key):
    """Create a composite image showing: original | annotated | info panel."""
    # Load images
    orig = Image.open(original_path).convert("RGB")
    annot_bytes = base64.b64decode(annotated_b64)
    annot = Image.open(io.BytesIO(annot_bytes)).convert("RGB")

    # Normalize sizes
    TARGET_H = 500
    orig_w = int(orig.width * TARGET_H / orig.height)
    annot_w = int(annot.width * TARGET_H / annot.height)
    orig = orig.resize((orig_w, TARGET_H), Image.LANCZOS)
    annot = annot.resize((annot_w, TARGET_H), Image.LANCZOS)

    # Info panel
    PANEL_W = 420
    BG_COLOR = (26, 27, 38)  # #1a1b26
    TEXT_COLOR = (224, 224, 224)
    ORANGE = (249, 115, 22)
    GREEN = (74, 222, 128)
    RED = (248, 113, 113)
    MUTED = (160, 160, 160)

    total_w = orig_w + annot_w + PANEL_W + 40  # 40 for margins
    card = Image.new("RGB", (total_w, TARGET_H + 60), BG_COLOR)
    draw = ImageDraw.Draw(card)

    # Try to load a decent font, fall back to default
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = font_title
        font_small = font_title
        font_tag = font_title

    # Title bar
    title = image_key.replace("_", " ").title()
    draw.text((15, 15), title, fill=ORANGE, font=font_title)

    # Paste images
    y_img = 50
    card.paste(orig, (10, y_img))
    card.paste(annot, (orig_w + 20, y_img))

    # Info panel
    px = orig_w + annot_w + 30
    py = y_img

    # Description
    draw.text((px, py), "DESCRIPTION", fill=ORANGE, font=font_tag)
    py += 22
    wrapped = textwrap.wrap(description, width=38)
    for line in wrapped[:6]:
        draw.text((px, py), line, fill=TEXT_COLOR, font=font_small)
        py += 17
    py += 10

    # Detected items
    draw.text((px, py), "DETECTED ITEMS", fill=ORANGE, font=font_tag)
    py += 22
    detected_names = [item["name"] for item in items]
    if detected_names:
        for name in detected_names:
            draw.text((px + 10, py), f"• {name}", fill=GREEN, font=font_body)
            py += 20
    else:
        draw.text((px + 10, py), "(no items detected)", fill=MUTED, font=font_body)
        py += 20
    py += 10

    # Ground truth
    draw.text((px, py), "GROUND TRUTH", fill=ORANGE, font=font_tag)
    py += 22
    for g in gt:
        # Check if this GT item was matched
        matched = any(fuzzy_match(d, g) for d in detected_names)
        color = GREEN if matched else RED
        symbol = "✓" if matched else "✗"
        draw.text((px + 10, py), f"{symbol} {g}", fill=color, font=font_body)
        py += 20
    py += 10

    # Scores
    if scores:
        draw.text((px, py), "SCORES", fill=ORANGE, font=font_tag)
        py += 22
        for k in ["health", "satiety", "bloat", "tasty", "addiction"]:
            v = scores.get(k, 0)
            bar_len = int(v * 15)
            bar = "█" * bar_len + "░" * (150 // 10 - bar_len)
            draw.text((px + 10, py), f"{k:>10}: {v}/10", fill=TEXT_COLOR, font=font_small)
            py += 17

    return card


# ============================================================
# MAIN TEST LOOP
# ============================================================
def run_tests():
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
        print(f"[{i}/17] Testing {key}")
        print(f"{'='*60}")

        t0 = time.time()
        with open(path, "rb") as f:
            resp = requests.post(URL, files={"image": (os.path.basename(path), f)})
        elapsed = time.time() - t0

        if resp.status_code != 200:
            print(f"  ERROR: status {resp.status_code}")
            continue

        data = resp.json()
        detected_items = data.get("items", [])
        detected_names = [item["name"] for item in detected_items]
        scores = data.get("scores", {})
        description = data.get("description", "")
        annotated_b64 = data.get("annotated_b64", "")

        print(f"  Detected: {detected_names}")
        print(f"  GT:       {gt}")
        print(f"  Time:     {elapsed:.1f}s")

        # Save annotated image
        if annotated_b64:
            annot_bytes = base64.b64decode(annotated_b64)
            annot_path = os.path.join(VIS_DIR, f"{key}_annotated.png")
            with open(annot_path, "wb") as af:
                af.write(annot_bytes)

        # Create and save visual result card
        card = create_result_card(path, annotated_b64, description, detected_items, scores, gt, key)
        card_path = os.path.join(VIS_DIR, f"{key}_card.png")
        card.save(card_path, quality=95)
        print(f"  Saved: {card_path}")

        results.append({
            "image": key,
            "ground_truth": gt,
            "gt_count": len(gt),
            "detected_items": detected_names,
            "detected_count": len(detected_names),
            "description": description,
            "scores": scores,
            "elapsed_s": round(elapsed, 1),
        })

    # Save JSON results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    return results


# ============================================================
# REVISED ANALYSIS (failure = wrong ID only, no detection = OK)
# ============================================================
def generate_revised_analysis(results):
    print("\n\nGenerating revised analysis figures...")

    # --- Style ---
    plt.rcParams.update({
        "figure.facecolor": "#1a1b26",
        "axes.facecolor": "#1a1b26",
        "axes.edgecolor": "#3b3d52",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#a0a0a0",
        "ytick.color": "#a0a0a0",
        "grid.color": "#2e3140",
        "font.family": "sans-serif",
        "font.size": 11,
    })
    ORANGE = "#f97316"
    BLUE = "#38bdf8"
    GREEN = "#4ade80"
    RED = "#f87171"
    YELLOW = "#fbbf24"
    PURPLE = "#c084fc"
    GRAY = "#6b7280"

    # --- Compute revised metrics per image ---
    per_image = []
    for r in results:
        gt = r["ground_truth"]
        det = r["detected_items"]

        gt_matched = [False] * len(gt)
        det_correct = [False] * len(det)

        for di, d in enumerate(det):
            for gi, g in enumerate(gt):
                if not gt_matched[gi] and fuzzy_match(d, g):
                    gt_matched[gi] = True
                    det_correct[di] = True
                    break

        correct = sum(det_correct)
        wrong = len(det) - correct  # WRONG identification = the only failure
        missed = len(gt) - sum(gt_matched)  # not a failure, just not detected
        no_output = len(det) == 0

        per_image.append({
            "image": r["image"],
            "gt_count": len(gt),
            "det_count": len(det),
            "correct": correct,
            "wrong": wrong,
            "missed": missed,
            "no_output": no_output,
            "elapsed_s": r["elapsed_s"],
            "scores": r["scores"],
        })

    labels = [m["image"].replace("sample_food_", "#") for m in per_image]
    x = np.arange(len(labels))

    # Aggregate
    total_correct = sum(m["correct"] for m in per_image)
    total_wrong = sum(m["wrong"] for m in per_image)
    total_detected = sum(m["det_count"] for m in per_image)
    total_gt = sum(m["gt_count"] for m in per_image)
    total_missed = sum(m["missed"] for m in per_image)
    images_with_output = sum(1 for m in per_image if not m["no_output"])
    images_no_output = sum(1 for m in per_image if m["no_output"])
    accuracy = total_correct / total_detected if total_detected > 0 else 0

    # ========== Figure 1: Per-image outcome breakdown ==========
    fig, ax = plt.subplots(figsize=(14, 6))
    correct_vals = [m["correct"] for m in per_image]
    wrong_vals = [m["wrong"] for m in per_image]
    missed_vals = [m["missed"] for m in per_image]

    ax.bar(x, correct_vals, label="Correct ID", color=GREEN, alpha=0.9)
    ax.bar(x, wrong_vals, bottom=correct_vals, label="Wrong ID (FAILURE)", color=RED, alpha=0.9)
    ax.bar(x, missed_vals, bottom=[c+w for c, w in zip(correct_vals, wrong_vals)],
           label="Not detected (OK)", color=GRAY, alpha=0.5)
    ax.set_xlabel("Sample Image")
    ax.set_ylabel("Count")
    ax.set_title("Per-Image Breakdown: Correct / Wrong / Not Detected",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_revised_breakdown.png"), dpi=150)
    plt.close()

    # ========== Figure 2: Accuracy per image (only images with detections) ==========
    fig, ax = plt.subplots(figsize=(14, 6))
    accs = []
    acc_labels = []
    acc_colors = []
    for m in per_image:
        if m["det_count"] > 0:
            acc = m["correct"] / m["det_count"]
            accs.append(acc)
            acc_labels.append(m["image"].replace("sample_food_", "#"))
            acc_colors.append(GREEN if acc >= 0.8 else YELLOW if acc >= 0.5 else RED)
        else:
            accs.append(None)
            acc_labels.append(m["image"].replace("sample_food_", "#"))
            acc_colors.append(GRAY)

    x2 = np.arange(len(acc_labels))
    for i, (a, c) in enumerate(zip(accs, acc_colors)):
        if a is not None:
            ax.bar(i, a, color=c, alpha=0.9)
        else:
            ax.bar(i, 0, color=GRAY, alpha=0.3)
            ax.text(i, 0.05, "N/A", ha="center", va="bottom", color=GRAY,
                    fontsize=9, fontstyle="italic")

    ax.axhline(y=accuracy, color=ORANGE, linestyle="--", linewidth=2,
               label=f"Overall: {accuracy:.0%}")
    ax.set_xlabel("Sample Image")
    ax.set_ylabel("Accuracy")
    ax.set_title("Identification Accuracy (Wrong ID = Failure, No Output = N/A)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels(acc_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_revised_accuracy.png"), dpi=150)
    plt.close()

    # ========== Figure 3: Revised aggregate dashboard ==========
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Pie: of all detections, how many were correct vs wrong
    ax = axes[0]
    sizes = [total_correct, total_wrong]
    pie_labels = [f"Correct: {total_correct}", f"Wrong: {total_wrong}"]
    pie_colors = [GREEN, RED]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=pie_labels, colors=pie_colors,
        autopct="%1.0f%%", startangle=90, pctdistance=0.6,
        textprops={"color": "#e0e0e0", "fontsize": 13})
    for t in autotexts:
        t.set_color("#1a1b26")
        t.set_fontweight("bold")
        t.set_fontsize(14)
    ax.set_title("When We Detect, Are We Right?", fontsize=13, fontweight="bold")

    # Pie: image-level outcomes
    ax = axes[1]
    # Categories: perfect (all correct, 0 wrong), partial (some correct, some wrong/missed),
    # no output (0 detections), all wrong
    perfect = sum(1 for m in per_image if m["det_count"] > 0 and m["wrong"] == 0)
    has_wrong = sum(1 for m in per_image if m["wrong"] > 0)
    no_out = images_no_output
    sizes2 = [perfect, has_wrong, no_out]
    labels2 = [f"Clean output: {perfect}", f"Has wrong ID: {has_wrong}", f"No output: {no_out}"]
    colors2 = [GREEN, RED, GRAY]
    # Filter out zeros
    filt = [(s, l, c) for s, l, c in zip(sizes2, labels2, colors2) if s > 0]
    if filt:
        sz, lb, cl = zip(*filt)
        wedges2, texts2, auto2 = ax.pie(
            sz, labels=lb, colors=cl, autopct="%1.0f%%", startangle=90,
            textprops={"color": "#e0e0e0", "fontsize": 12})
        for t in auto2:
            t.set_color("#1a1b26")
            t.set_fontweight("bold")
    ax.set_title("Image-Level Outcomes (17 images)", fontsize=13, fontweight="bold")

    # Summary stats
    ax = axes[2]
    ax.axis("off")
    summary = (
        f"Total Images:          {len(per_image)}\n"
        f"  With output:         {images_with_output}\n"
        f"  No output (OK):      {images_no_output}\n"
        f"\n"
        f"Total Detections:      {total_detected}\n"
        f"  Correct:             {total_correct}\n"
        f"  Wrong (FAILURES):    {total_wrong}\n"
        f"\n"
        f"Identification Accuracy:\n"
        f"  {accuracy:.0%}  ({total_correct}/{total_detected})\n"
        f"\n"
        f"Coverage:\n"
        f"  {total_correct}/{total_gt} GT items found\n"
        f"  {total_missed} items not detected (OK)\n"
        f"\n"
        f"Avg Latency: {np.mean([m['elapsed_s'] for m in per_image]):.1f}s"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=13,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2e3140", edgecolor="#3b3d52"))
    ax.set_title("Revised Summary", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "09_revised_aggregate.png"), dpi=150)
    plt.close()

    # ========== Figure 4: Latency vs detection count scatter ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    det_counts = [m["det_count"] for m in per_image]
    latencies = [m["elapsed_s"] for m in per_image]
    gt_counts = [m["gt_count"] for m in per_image]

    scatter = ax.scatter(det_counts, latencies, c=[m["correct"]/m["det_count"] if m["det_count"] > 0 else 0.5
                         for m in per_image],
                         cmap="RdYlGn", s=100, edgecolors="white", linewidth=1.5,
                         vmin=0, vmax=1, zorder=5)
    for i, m in enumerate(per_image):
        ax.annotate(m["image"].replace("sample_food_", "#"),
                    (m["det_count"], m["elapsed_s"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color="#a0a0a0")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Accuracy", color="#e0e0e0")
    cbar.ax.yaxis.set_tick_params(color="#a0a0a0")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#a0a0a0")
    ax.set_xlabel("Number of Detections")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Latency vs Detections (color = accuracy)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_latency_vs_detections.png"), dpi=150)
    plt.close()

    # ========== Figure 5: Average radar ==========
    score_keys = ["health", "satiety", "bloat", "tasty", "addiction"]
    avg_scores = {k: np.mean([m["scores"].get(k, 0) for m in per_image]) for k in score_keys}

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(score_keys), endpoint=False).tolist()
    values = [avg_scores[k] for k in score_keys]
    angles += angles[:1]
    values += values[:1]
    ax.fill(angles, values, color=ORANGE, alpha=0.25)
    ax.plot(angles, values, color=ORANGE, linewidth=2, marker="o", markersize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([k.capitalize() for k in score_keys], fontsize=13, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_title("Average Nutritional Scores\n(across 17 samples)", fontsize=14, fontweight="bold", pad=20)
    ax.grid(color="#3b3d52")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "11_avg_radar.png"), dpi=150)
    plt.close()

    print(f"\nAll figures saved to {FIG_DIR}/")

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"REVISED ANALYSIS (wrong ID = failure, no output = OK)")
    print(f"{'='*60}")
    print(f"Identification Accuracy: {accuracy:.0%} ({total_correct}/{total_detected})")
    print(f"Images with output:      {images_with_output}/17")
    print(f"Images with 0 wrong IDs: {perfect}/{images_with_output} (of those with output)")
    print(f"Total wrong IDs:         {total_wrong}")
    print(f"Coverage:                {total_correct}/{total_gt} GT items found")


if __name__ == "__main__":
    results = run_tests()
    generate_revised_analysis(results)
    print(f"\nVisual result cards saved to: {VIS_DIR}/")
    print(f"Analysis figures saved to:    {FIG_DIR}/")
