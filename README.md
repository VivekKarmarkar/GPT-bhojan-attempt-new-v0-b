# What's on your Plate?

**GPT-4o + YOLOv8 + SAM — LLM-driven food analysis**

Upload a photo of your meal. Three AI models — the Brain, the Eyes, and the Scissors — work together to find every item on your plate, cut each one out at the pixel level, and score the meal on five dimensions: health, satiety, bloat, tastiness, and addiction potential.

## How It Works

The pipeline coordinates three models through nine steps:

| Step | Model | What it does |
|------|-------|-------------|
| 1. Upload Photo | — | User provides a meal image |
| 2. Analyze Plate | Brain (GPT-4o) | Names every dish, rates the meal on five scales |
| 3. Draw Boxes | Eyes (YOLOv8) | Bounding boxes around every object found |
| 4. Classify Each Box | Brain (GPT-4o) | Visual judgment — food item or junk? |
| 5. Remove Duplicates | Logic | Same label + >50% overlap → keep tighter box |
| 6. Cut Precise Shapes | Scissors (SAM) | Pixel-level outlines, 3 candidates per box, best score wins |
| 7. Verify A vs B | Brain (GPT-4o) | Cut-out vs. complement — did the Scissors trace the right thing? |
| 8. Brightness Check | Logic | <100 bright pixels → drop the item |
| 9. Color & Label | Logic | Overlay masks, bounding boxes, and labels on the original image |

Steps 2 and 3 run in parallel — neither depends on the other. Both converge at Step 4.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash setup.sh          # downloads SAM checkpoint (~375 MB)
```

Create a `.env` file:

```
OPENAI_API_KEY=your-key-here
```

## Run

```bash
python app.py
```

Opens at `http://localhost:8080`. Upload a food photo and the pipeline returns an annotated image, a five-score radar chart, and an item list.

## Project Structure

```
app.py           Flask routes, model loading at startup
config.py        Device detection (CUDA/CPU), constants, env loading
prompts.py       All GPT-4o system prompts
analysis.py      GPT-4o API calls (plate analysis, classification, verification)
vision.py        YOLO + SAM pipeline: detect → classify → dedup → segment → verify → overlay

templates/       Single-page frontend
static/          Dark theme CSS + vanilla JS (Chart.js radar)
eval/            Test suite, figures, and evaluation scripts
Sample Food Library/   17 test images + ground truth labels
project website/       Static project website
```

## Evaluation

Tested against a 17-image suite of Indian meals:

- **90% identification accuracy** — 28/31 correct detections
- 11 of 14 images with output had zero wrong identifications
- 3 misidentifications all followed the same pattern: the Brain defaulted to "Sambar" for unrecognized curry bowls
- 3 images produced no output (YOLO found no bounding boxes)

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o with vision)
- CUDA GPU recommended (CPU fallback supported)
