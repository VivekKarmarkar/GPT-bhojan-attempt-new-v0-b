"""Generate analysis figures from test_results.json."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "test_results.json")
FIG_DIR = os.path.join(os.path.dirname(__file__), "test_figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(RESULTS_PATH) as f:
    results = json.load(f)

# ---------- Helpers ----------

def fuzzy_match(detected, gt):
    """Check if a detected label fuzzy-matches a ground truth label."""
    d = detected.lower().strip()
    g = gt.lower().strip()
    # Direct substring match
    if d in g or g in d:
        return True
    # Key word overlap
    d_words = set(d.replace("/", " ").replace("-", " ").split())
    g_words = set(g.replace("/", " ").replace("-", " ").split())
    # At least one meaningful word in common (skip short words)
    meaningful = d_words.intersection(g_words) - {"ka", "a", "the", "or", "and", "with"}
    return len(meaningful) > 0

def compute_metrics(result):
    """Compute per-image precision, recall, and match details."""
    gt = result.get("ground_truth", [])
    det = result.get("detected_items", [])
    if not gt:
        return {"precision": 1.0, "recall": 1.0, "tp": 0, "fp": 0, "fn": 0}

    gt_matched = [False] * len(gt)
    det_matched = [False] * len(det)

    for di, d in enumerate(det):
        for gi, g in enumerate(gt):
            if not gt_matched[gi] and fuzzy_match(d, g):
                gt_matched[gi] = True
                det_matched[di] = True
                break

    tp = sum(det_matched)
    fp = len(det) - tp
    fn = len(gt) - sum(gt_matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


# ---------- Compute all metrics ----------

metrics = []
for r in results:
    m = compute_metrics(r)
    m["image"] = r["image"]
    m["gt_count"] = r.get("gt_count", len(r.get("ground_truth", [])))
    m["det_count"] = r.get("detected_count", len(r.get("detected_items", [])))
    m["elapsed_s"] = r.get("elapsed_s", 0)
    m["scores"] = r.get("scores", {})
    metrics.append(m)

# ---------- Style ----------
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
PURPLE = "#c084fc"

# ========== Figure 1: Detection Count (GT vs Detected) ==========
fig, ax = plt.subplots(figsize=(14, 6))
labels = [m["image"].replace("sample_food_", "#") for m in metrics]
x = np.arange(len(labels))
w = 0.35
bars_gt = ax.bar(x - w/2, [m["gt_count"] for m in metrics], w, label="Ground Truth", color=BLUE, alpha=0.85)
bars_det = ax.bar(x + w/2, [m["det_count"] for m in metrics], w, label="Detected", color=ORANGE, alpha=0.85)
ax.set_xlabel("Sample Image")
ax.set_ylabel("Item Count")
ax.set_title("Ground Truth vs Detected Item Counts", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
ax.set_ylim(0, max(max(m["gt_count"] for m in metrics), max(m["det_count"] for m in metrics)) + 1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_gt_vs_detected_counts.png"), dpi=150)
plt.close()

# ========== Figure 2: Precision & Recall per image ==========
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w/2, [m["precision"] for m in metrics], w, label="Precision", color=GREEN, alpha=0.85)
ax.bar(x + w/2, [m["recall"] for m in metrics], w, label="Recall", color=PURPLE, alpha=0.85)
ax.set_xlabel("Sample Image")
ax.set_ylabel("Score (0-1)")
ax.set_title("Precision & Recall per Image (Fuzzy Label Matching)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
ax.set_ylim(0, 1.15)
ax.axhline(y=1.0, color="#555", linestyle="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_precision_recall.png"), dpi=150)
plt.close()

# ========== Figure 3: TP / FP / FN breakdown ==========
fig, ax = plt.subplots(figsize=(14, 6))
tp = [m["tp"] for m in metrics]
fp = [m["fp"] for m in metrics]
fn = [m["fn"] for m in metrics]
ax.bar(x, tp, label="True Positive", color=GREEN, alpha=0.85)
ax.bar(x, fp, bottom=tp, label="False Positive", color=RED, alpha=0.85)
ax.bar(x, fn, bottom=[t+f for t, f in zip(tp, fp)], label="False Negative (missed)", color="#fbbf24", alpha=0.85)
ax.set_xlabel("Sample Image")
ax.set_ylabel("Count")
ax.set_title("Detection Breakdown: TP / FP / FN", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_tp_fp_fn.png"), dpi=150)
plt.close()

# ========== Figure 4: Latency per image ==========
fig, ax = plt.subplots(figsize=(14, 6))
times = [m["elapsed_s"] for m in metrics]
colors_lat = [ORANGE if t > 30 else BLUE for t in times]
ax.bar(x, times, color=colors_lat, alpha=0.85)
ax.axhline(y=np.mean(times), color=GREEN, linestyle="--", linewidth=2, label=f"Mean: {np.mean(times):.1f}s")
ax.set_xlabel("Sample Image")
ax.set_ylabel("Time (seconds)")
ax.set_title("End-to-End Latency per Image", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_latency.png"), dpi=150)
plt.close()

# ========== Figure 5: Aggregate summary ==========
total_tp = sum(tp)
total_fp = sum(fp)
total_fn = sum(fn)
total_gt = sum(m["gt_count"] for m in metrics)
total_det = sum(m["det_count"] for m in metrics)
macro_prec = np.mean([m["precision"] for m in metrics if m["gt_count"] > 0])
macro_rec = np.mean([m["recall"] for m in metrics if m["gt_count"] > 0])
micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Pie: detection outcome
ax = axes[0]
sizes = [total_tp, total_fp, total_fn]
pie_labels = [f"TP: {total_tp}", f"FP: {total_fp}", f"FN: {total_fn}"]
pie_colors = [GREEN, RED, "#fbbf24"]
wedges, texts, autotexts = ax.pie(sizes, labels=pie_labels, colors=pie_colors,
                                   autopct="%1.0f%%", startangle=90,
                                   textprops={"color": "#e0e0e0", "fontsize": 12})
for t in autotexts:
    t.set_color("#1a1b26")
    t.set_fontweight("bold")
ax.set_title("Detection Outcomes", fontsize=13, fontweight="bold")

# Bar: macro vs micro precision/recall
ax = axes[1]
bar_x = np.arange(2)
ax.bar(bar_x - 0.15, [macro_prec, micro_prec], 0.3, label="Precision", color=GREEN, alpha=0.85)
ax.bar(bar_x + 0.15, [macro_rec, micro_rec], 0.3, label="Recall", color=PURPLE, alpha=0.85)
ax.set_xticks(bar_x)
ax.set_xticklabels(["Macro Avg", "Micro Avg"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Aggregate Precision & Recall", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Text: summary stats
ax = axes[2]
ax.axis("off")
summary_text = (
    f"Total Images: {len(metrics)}\n"
    f"Total GT Items: {total_gt}\n"
    f"Total Detected: {total_det}\n"
    f"\n"
    f"True Positives:  {total_tp}\n"
    f"False Positives: {total_fp}\n"
    f"False Negatives: {total_fn}\n"
    f"\n"
    f"Macro Precision: {macro_prec:.2f}\n"
    f"Macro Recall:    {macro_rec:.2f}\n"
    f"Micro Precision: {micro_prec:.2f}\n"
    f"Micro Recall:    {micro_rec:.2f}\n"
    f"\n"
    f"Avg Latency: {np.mean(times):.1f}s\n"
    f"Median Latency: {np.median(times):.1f}s"
)
ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=13,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2e3140", edgecolor="#3b3d52"))
ax.set_title("Summary Stats", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_aggregate_summary.png"), dpi=150)
plt.close()

# ========== Figure 6: Score radar (average across all images) ==========
score_keys = ["health", "satiety", "bloat", "tasty", "addiction"]
avg_scores = {k: np.mean([m["scores"].get(k, 0) for m in metrics]) for k in score_keys}

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
plt.savefig(os.path.join(FIG_DIR, "06_avg_radar.png"), dpi=150)
plt.close()

print(f"All figures saved to {FIG_DIR}/")
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
