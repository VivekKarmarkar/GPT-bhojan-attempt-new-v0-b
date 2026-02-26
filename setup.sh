#!/usr/bin/env bash
# setup.sh — Download SAM checkpoint and verify environment
set -e

MODEL_DIR="models"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILE="$MODEL_DIR/sam_vit_b_01ec64.pth"

mkdir -p "$MODEL_DIR"

if [ -f "$SAM_FILE" ]; then
    echo "SAM checkpoint already exists at $SAM_FILE"
else
    echo "Downloading SAM ViT-B checkpoint (~375 MB)..."
    wget -q --show-progress -O "$SAM_FILE" "$SAM_URL"
    echo "Download complete: $SAM_FILE"
fi

# Verify .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: No .env file found."
    echo "Copy .env.example to .env and add your OpenAI API key:"
    echo "  cp .env.example .env"
fi

echo ""
echo "Setup complete. Run the app with:"
echo "  source venv/bin/activate"
echo "  python app.py"
