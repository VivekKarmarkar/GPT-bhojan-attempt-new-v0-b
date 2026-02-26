# GPT Bhojan — Project Motivation

## The Portfolio Sweet Spot

This project sits at the intersection of three AI capabilities that rarely appear together in a single workflow:

1. **Brain (GPT-4o)** — Language model as orchestrator. Analyzes the full plate, classifies individual crops, and verifies segmentation masks. Three distinct GPT calls per food item, each with a different purpose.

2. **Eyes (YOLOv8)** — Object detection provides the spatial awareness. YOLO finds bounding boxes around objects on the plate, giving GPT something to reason about at the item level rather than the whole-image level.

3. **Scissors (SAM)** — Segment Anything Model cuts out pixel-precise masks using YOLO's bounding boxes as prompts. The A/B verification step (GPT choosing between mask and its complement) compensates for SAM's occasional mistakes.

## Why Food?

Food is the perfect domain for demonstrating this pipeline:

- **Visually complex** — overlapping items, sauces, garnishes make segmentation non-trivial
- **Culturally rich** — Indian cuisine has hundreds of dishes, and GPT's world knowledge can identify them from visual cues alone
- **Universally relatable** — everyone eats; the demo needs no domain expertise to appreciate
- **Multi-item scenes** — a typical Indian plate has 3-7 distinct items, exercising the full detection-segmentation-classification loop

## What This Demonstrates

For a portfolio reviewer or interviewer, this project shows:

- **LLM-as-orchestrator pattern** — GPT doesn't just answer questions; it makes decisions that drive a multi-model pipeline
- **Multi-model coordination** — three AI models (GPT-4o, YOLOv8, SAM) work in sequence, each compensating for the others' weaknesses
- **Practical GPU management** — auto-detecting CUDA, loading heavy models once, running inference per-item
- **Clean API design** — Flask backend returns structured JSON; vanilla JS frontend renders it
- **Error compensation** — the A/B verification pattern is a real engineering solution to a real segmentation failure mode

## From Monolith to Portfolio

The original (`gpt_bhojan_app_v12.py`) was a 338-line Streamlit monolith with hardcoded Windows paths, forced CUDA, fragile regex parsing, and Supabase logging. The rebuild:

- Eliminates Streamlit and Supabase dependencies
- Replaces regex parsing with GPT-4o JSON mode
- Auto-detects GPU with CPU fallback
- Separates concerns into 4 focused Python files
- Adds a dark-themed portfolio UI with radar chart visualization

The goal: a localhost demo you can screen-record in 60 seconds and drop into a portfolio.
