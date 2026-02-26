# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT Bhojan v0-b is a Flask-based food image analysis portfolio project focused on Indian cuisine. Users upload a food photo via a dark-themed web UI and receive: a GPT-4o-powered 5-score analysis (health, satiety, bloat, tasty, addiction), YOLOv8 object detection, SAM segmentation with GPT A/B verification, and an annotated overlay image with a radar chart visualization.

## Running the App

```bash
# First-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash setup.sh          # downloads SAM checkpoint (~375 MB)
cp .env.example .env   # add your OPENAI_API_KEY

# Run
python app.py          # starts on http://localhost:8080
```

Requires `.env` with:
- `OPENAI_API_KEY` — GPT-4o access

## Architecture

Five Python modules + a vanilla JS frontend:

```
config.py      — Device detection (CUDA/CPU), constants, env loading
prompts.py     — Single source of truth for all GPT-4o system prompts
analysis.py    — GPT-4o API calls: full-plate analysis, crop classification, A/B mask verification
vision.py      — YOLO + SAM pipeline: detect → classify → dedup → segment → verify → overlay
app.py         — Flask routes (GET /, POST /analyze), model loading at startup
```

```
Image Upload → base64 encode
    → GPT-4o (JSON mode: description, items, 5 scores)
    → YOLOv8 detection (bounding boxes)
    → Pass 1 — GPT classification:
        → Cost guard (skip boxes >90% of image)
        → GPT-4o visual classification (match crop to food items)
    → Dedup: same label + IoU > 0.5 → keep tighter box
    → Pass 2 — SAM segmentation + verification:
        → SAM set_image() once
        → For each surviving box:
            → SAM segmentation (box-prompted, multimask, best score)
            → GPT-4o A/B verification (mask vs complement)
            → Brightness check (>30 threshold, >100 pixels)
            → Alpha blend overlay (0.6) + bbox + label
    → JSON response: {annotated_b64, description, scores, items}
    → Frontend: annotated image + radar chart + item pills
```

## Key Dependencies

- **Flask** — Web framework, two routes
- **OpenAI SDK** — GPT-4o with vision (JSON response mode)
- **ultralytics** — YOLOv8m for object detection (auto-downloads `yolov8m.pt`)
- **segment-anything** — Meta's SAM (ViT-B) for pixel-level segmentation
- **torch + torchvision** — GPU inference (CUDA auto-detected, CPU fallback)
- **OpenCV + PIL + numpy** — Image processing pipeline
- **Chart.js** (CDN) — Radar chart in frontend

## File Structure

```
GPT-bhojan-attempt-new-v0-b/
│
├── app.py                      Flask routes, model loading at startup
├── config.py                   Device detection, constants, env loading
├── prompts.py                  All GPT-4o system prompts (single source of truth)
├── analysis.py                 GPT-4o API calls (imports prompts from prompts.py)
├── vision.py                   YOLO + SAM pipeline, IoU dedup, overlay drawing
├── setup.sh                    SAM checkpoint download script
├── requirements.txt
├── .env / .env.example
├── yolov8m.pt                  YOLO weights (auto-downloaded)
│
├── templates/
│   └── index.html              Single-page frontend
├── static/
│   ├── main.js                 Upload, Chart.js radar, DOM updates
│   └── style.css               Dark theme styles
│
├── eval/                       Testing & evaluation (self-contained)
│   ├── run_test_suite.py           Basic API test → saves JSON
│   ├── run_visual_test.py          Full test + visual card generation
│   ├── regenerate_cards.py         Regenerate cards from saved data (no API calls)
│   ├── generate_figures.py         Original analysis figures
│   ├── test_results.json           Raw results data (17 images)
│   ├── test_figures/               Analysis charts (11 PNGs)
│   └── visual_results/            17 annotated images + 17 result cards
│
├── media/
│   ├── results/                Standalone figures (e.g. simple_piechart.png)
│   ├── music/                  Background music assets
│   └── showcase examples/      Hand-picked showcase cards
│
├── Sample Food Library/        17 test images + ground_truth_labels.md
├── reference/                  Old codebase for reference
│   └── gpt_bhojan_app_v12.py      Original Streamlit monolith
│
├── models/                     SAM checkpoint (~375 MB)
└── info/                       Project documentation
    └── CLAUDE.md
```

## Evaluation Results (17-image test suite)

- **Identification Accuracy: 90%** (28/31 detections correct)
- **3 wrong IDs** — all the same "Sambar bias" pattern (GPT defaults to "Sambar" for unrecognized curry bowls)
- **3 images with no output** — YOLO found no boxes (Modak, Salmon, Rasmalai) — not counted as failures
- **11/14 images with output** had zero wrong identifications
- Run eval scripts: `cd eval && python run_visual_test.py` (re-runs full pipeline) or `python regenerate_cards.py` (regenerates cards from saved data only)

## Sibling Project

`../GPT-bhojan-attempt-new-v1/` contains a more advanced version using a ViT-Small/16 + Claude Opus cascade classifier for 80 Indian food classes, with training infrastructure, a Flask inference UI, and a live dashboard.
