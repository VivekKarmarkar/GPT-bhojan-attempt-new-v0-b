"""Regenerate visual result cards and analysis figures from saved data.
No pipeline re-run — reads test_results.json + saved annotated images."""
import json
import os
import base64
import io
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Sample Food Library")
VIS_DIR = os.path.join(os.path.dirname(__file__), "visual_results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "test_figures")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "test_results.json")

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

# ---- Semantic equivalences for rule 3 (reasonable overlap) ----
# Each tuple: (word_a, word_b) means these are interchangeable
EQUIVALENCES = [
    ("vada", "wada"),
    ("subji", "sabzi"),
    ("subji", "stir"),      # "stir-fry" normalizes to words "stir" + "fry"
    ("sabzi", "stir"),
    ("gajar", "carrot"),
    ("naan", "nan"),
    ("roti", "chapati"),
    ("paratha", "parantha"),
    ("toast", "sandwich"),
    ("open-faced", "toast"),
    ("curd", "raita"),
    ("halwa", "halva"),
]

def _normalize(s):
    return s.lower().strip().replace("-", " ").replace("/", " ")

def _words(s):
    return set(_normalize(s).split())

def _expand_equivalences(words):
    """Expand a word set with all known equivalences."""
    expanded = set(words)
    for w in list(words):
        for a, b in EQUIVALENCES:
            if w == a:
                expanded.add(b)
            elif w == b:
                expanded.add(a)
    return expanded


def fuzzy_match(detected, gt):
    """Three-rule matching:
    1. Detected is a substring of GT (detected is a subset label)
    2. GT is a substring of detected (GT is a subset label)
    3. Reasonable semantic overlap (shared words after equivalence expansion)
    """
    d_norm = _normalize(detected)
    g_norm = _normalize(gt)

    # Rule 1: detected is substring of GT
    if d_norm in g_norm:
        return True
    # Rule 2: GT is substring of detected
    if g_norm in d_norm:
        return True

    # Rule 3: word-level overlap with equivalence expansion
    d_words = _expand_equivalences(_words(detected))
    g_words = _expand_equivalences(_words(gt))
    skip = {"ka", "a", "the", "or", "and", "with", "of", "in"}
    meaningful_d = d_words - skip
    meaningful_g = g_words - skip
    overlap = meaningful_d.intersection(meaningful_g)
    if len(overlap) > 0:
        return True

    return False


def compute_matches(gt_list, det_list):
    """Returns (gt_matched[bool], det_matched[bool]) arrays."""
    gt_matched = [False] * len(gt_list)
    det_matched = [False] * len(det_list)
    for di, d in enumerate(det_list):
        for gi, g in enumerate(gt_list):
            if not gt_matched[gi] and fuzzy_match(d, g):
                gt_matched[gi] = True
                det_matched[di] = True
                break
    return gt_matched, det_matched


# ---- Card generation ----

def create_result_card(original_path, annotated_path, description, detected_names,
                       scores, gt, gt_matched, det_matched, image_key):
    orig = Image.open(original_path).convert("RGB")
    annot = Image.open(annotated_path).convert("RGB")

    TARGET_H = 500
    orig_w = int(orig.width * TARGET_H / orig.height)
    annot_w = int(annot.width * TARGET_H / annot.height)
    orig = orig.resize((orig_w, TARGET_H), Image.LANCZOS)
    annot = annot.resize((annot_w, TARGET_H), Image.LANCZOS)

    PANEL_W = 420
    BG = (26, 27, 38)
    TEXT = (224, 224, 224)
    ORANGE = (249, 115, 22)
    GREEN = (74, 222, 128)
    RED = (248, 113, 113)
    MUTED = (160, 160, 160)
    GRAY_TAG = (107, 114, 128)

    total_w = orig_w + annot_w + PANEL_W + 40
    card = Image.new("RGB", (total_w, TARGET_H + 60), BG)
    draw = ImageDraw.Draw(card)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font_title = font_body = font_small = font_tag = ImageFont.load_default()

    # Title
    title = image_key.replace("_", " ").title()
    draw.text((15, 15), title, fill=ORANGE, font=font_title)

    y_img = 50
    card.paste(orig, (10, y_img))
    card.paste(annot, (orig_w + 20, y_img))

    px = orig_w + annot_w + 30
    py = y_img

    # Description
    draw.text((px, py), "DESCRIPTION", fill=ORANGE, font=font_tag)
    py += 22
    for line in textwrap.wrap(description, width=38)[:6]:
        draw.text((px, py), line, fill=TEXT, font=font_small)
        py += 17
    py += 10

    # Detected items with match status
    draw.text((px, py), "DETECTED ITEMS", fill=ORANGE, font=font_tag)
    py += 22
    if detected_names:
        for i, name in enumerate(detected_names):
            matched = det_matched[i] if i < len(det_matched) else False
            color = GREEN if matched else RED
            symbol = "+" if matched else "x"
            draw.text((px + 10, py), f"{symbol} {name}", fill=color, font=font_body)
            py += 20
    else:
        draw.text((px + 10, py), "(no detections — OK)", fill=MUTED, font=font_body)
        py += 20
    py += 10

    # Ground truth with match status
    draw.text((px, py), "GROUND TRUTH", fill=ORANGE, font=font_tag)
    py += 22
    for i, g in enumerate(gt):
        matched = gt_matched[i] if i < len(gt_matched) else False
        if matched:
            color = GREEN
            symbol = "+"
        else:
            color = GRAY_TAG
            symbol = "-"  # missed, not a failure
        draw.text((px + 10, py), f"{symbol} {g}", fill=color, font=font_body)
        py += 20
    py += 10

    # Scores
    if scores:
        draw.text((px, py), "SCORES", fill=ORANGE, font=font_tag)
        py += 22
        for k in ["health", "satiety", "bloat", "tasty", "addiction"]:
            v = scores.get(k, 0)
            draw.text((px + 10, py), f"{k:>10}: {v}/10", fill=TEXT, font=font_small)
            py += 17

    return card


def generate_figures(per_image):
    plt.rcParams.update({
        "figure.facecolor": "#1a1b26", "axes.facecolor": "#1a1b26",
        "axes.edgecolor": "#3b3d52", "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0", "xtick.color": "#a0a0a0",
        "ytick.color": "#a0a0a0", "grid.color": "#2e3140",
        "font.family": "sans-serif", "font.size": 11,
    })
    GREEN = "#4ade80"
    RED = "#f87171"
    ORANGE = "#f97316"
    YELLOW = "#fbbf24"
    GRAY = "#6b7280"

    labels = [m["image"].replace("sample_food_", "#") for m in per_image]
    x = np.arange(len(labels))

    total_correct = sum(m["correct"] for m in per_image)
    total_wrong = sum(m["wrong"] for m in per_image)
    total_detected = sum(m["det_count"] for m in per_image)
    total_gt = sum(m["gt_count"] for m in per_image)
    images_with_output = sum(1 for m in per_image if m["det_count"] > 0)
    images_no_output = sum(1 for m in per_image if m["det_count"] == 0)
    accuracy = total_correct / total_detected if total_detected > 0 else 0

    # Fig 1: Per-image breakdown
    fig, ax = plt.subplots(figsize=(14, 6))
    c = [m["correct"] for m in per_image]
    w = [m["wrong"] for m in per_image]
    mi = [m["missed"] for m in per_image]
    ax.bar(x, c, label="Correct ID", color=GREEN, alpha=0.9)
    ax.bar(x, w, bottom=c, label="Wrong ID (FAILURE)", color=RED, alpha=0.9)
    ax.bar(x, mi, bottom=[a+b for a, b in zip(c, w)], label="Not detected (OK)", color=GRAY, alpha=0.5)
    ax.set_xlabel("Sample Image"); ax.set_ylabel("Count")
    ax.set_title("Per-Image Breakdown: Correct / Wrong / Not Detected", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_revised_breakdown.png"), dpi=150)
    plt.close()

    # Fig 2: Accuracy per image
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(per_image):
        if m["det_count"] > 0:
            acc = m["correct"] / m["det_count"]
            color = GREEN if acc >= 0.8 else YELLOW if acc >= 0.5 else RED
            ax.bar(i, acc, color=color, alpha=0.9)
        else:
            ax.bar(i, 0, color=GRAY, alpha=0.3)
            ax.text(i, 0.05, "N/A", ha="center", va="bottom", color=GRAY, fontsize=9, fontstyle="italic")
    ax.axhline(y=accuracy, color=ORANGE, linestyle="--", linewidth=2, label=f"Overall: {accuracy:.0%}")
    ax.set_xlabel("Sample Image"); ax.set_ylabel("Accuracy")
    ax.set_title("Identification Accuracy (Wrong ID = Failure, No Output = N/A)", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.15); ax.legend(fontsize=12); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_revised_accuracy.png"), dpi=150)
    plt.close()

    # Fig 3: Aggregate dashboard
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        [total_correct, total_wrong],
        labels=[f"Correct: {total_correct}", f"Wrong: {total_wrong}"],
        colors=[GREEN, RED], autopct="%1.0f%%", startangle=90, pctdistance=0.6,
        textprops={"color": "#e0e0e0", "fontsize": 13})
    for t in autotexts: t.set_color("#1a1b26"); t.set_fontweight("bold"); t.set_fontsize(14)
    ax.set_title("When We Detect, Are We Right?", fontsize=13, fontweight="bold")

    ax = axes[1]
    perfect = sum(1 for m in per_image if m["det_count"] > 0 and m["wrong"] == 0)
    has_wrong = sum(1 for m in per_image if m["wrong"] > 0)
    filt = [(s, l, c) for s, l, c in
            [(perfect, f"Clean output: {perfect}", GREEN),
             (has_wrong, f"Has wrong ID: {has_wrong}", RED),
             (images_no_output, f"No output: {images_no_output}", GRAY)] if s > 0]
    sz, lb, cl = zip(*filt)
    wedges2, texts2, auto2 = ax.pie(sz, labels=lb, colors=cl, autopct="%1.0f%%", startangle=90,
                                     textprops={"color": "#e0e0e0", "fontsize": 12})
    for t in auto2: t.set_color("#1a1b26"); t.set_fontweight("bold")
    ax.set_title("Image-Level Outcomes (17 images)", fontsize=13, fontweight="bold")

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
        f"  {total_gt - total_correct} not detected (OK)\n"
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

    # Fig 4: Latency scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        [m["det_count"] for m in per_image], [m["elapsed_s"] for m in per_image],
        c=[m["correct"]/m["det_count"] if m["det_count"] > 0 else 0.5 for m in per_image],
        cmap="RdYlGn", s=100, edgecolors="white", linewidth=1.5, vmin=0, vmax=1, zorder=5)
    for m in per_image:
        ax.annotate(m["image"].replace("sample_food_", "#"),
                    (m["det_count"], m["elapsed_s"]), textcoords="offset points",
                    xytext=(8, 4), fontsize=9, color="#a0a0a0")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Accuracy", color="#e0e0e0")
    cbar.ax.yaxis.set_tick_params(color="#a0a0a0")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#a0a0a0")
    ax.set_xlabel("Number of Detections"); ax.set_ylabel("Latency (seconds)")
    ax.set_title("Latency vs Detections (color = accuracy)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "10_latency_vs_detections.png"), dpi=150)
    plt.close()


def find_images():
    files = os.listdir(SAMPLE_DIR)
    mapping = {}
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            mapping[os.path.splitext(f)[0]] = os.path.join(SAMPLE_DIR, f)
    return mapping


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    images = find_images()
    per_image = []

    for r in results:
        key = r["image"]
        gt = GROUND_TRUTH.get(key, [])
        det = r["detected_items"]
        gt_matched, det_matched = compute_matches(gt, det)

        correct = sum(det_matched)
        wrong = len(det) - correct
        missed = len(gt) - sum(gt_matched)

        print(f"{key}: det={det} | correct={correct} wrong={wrong} missed={missed}")
        if wrong > 0:
            for i, d in enumerate(det):
                if not det_matched[i]:
                    print(f"  WRONG: '{d}' has no GT match")

        per_image.append({
            "image": key, "gt_count": len(gt), "det_count": len(det),
            "correct": correct, "wrong": wrong, "missed": missed,
            "elapsed_s": r["elapsed_s"], "scores": r["scores"],
        })

        # Regenerate card
        orig_path = images.get(key)
        annot_path = os.path.join(VIS_DIR, f"{key}_annotated.png")
        if orig_path and os.path.exists(annot_path):
            card = create_result_card(
                orig_path, annot_path, r["description"], det,
                r["scores"], gt, gt_matched, det_matched, key)
            card.save(os.path.join(VIS_DIR, f"{key}_card.png"), quality=95)

    # Regenerate figures
    generate_figures(per_image)

    total_det = sum(m["det_count"] for m in per_image)
    total_correct = sum(m["correct"] for m in per_image)
    total_wrong = sum(m["wrong"] for m in per_image)
    acc = total_correct / total_det if total_det > 0 else 0

    print(f"\n{'='*50}")
    print(f"Accuracy: {acc:.0%} ({total_correct}/{total_det})")
    print(f"Wrong IDs: {total_wrong}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
