# GPT Bhojan v0-b — Complete Code Explanation

A plain-language walkthrough of every phase in `gpt_bhojan_app_v12.py`.

---

## Phase 0: Setup (Lines 1–41)

**Imports** load 14 libraries. The heavy ones are:
- `streamlit` — the web UI framework (every `st.something()` draws something in the browser)
- `openai` — talks to GPT-4 Turbo (the **"brain"**)
- `ultralytics` — runs YOLOv8 (the **"eyes"** that find objects)
- `segment_anything` — runs SAM (the **"scissors"** that cut out precise shapes)
- `supabase` — cloud database + file storage
- `cv2` (OpenCV), `PIL`, `numpy`, `matplotlib` — image processing tools

**API clients** are created using secret keys stored in Streamlit's secrets file (never in code). One client for OpenAI, one for Supabase.

**Two local folders** are created if they don't exist: `food_library/` (for individual segmented food items) and `favorite_meals/` (for bookmarked plates).

---

## Phase 1: Image Upload (Lines 43–64)

When the user picks a photo:

1. **Timestamp** is recorded (`2025-06-02T14:35:22...`)
2. **A placeholder** (`st.empty()`) reserves a spot on screen — this lets the app swap images in the same location later
3. The image is **displayed** in that placeholder
4. The image bytes are **base64-encoded** — converted from binary into a long text string, because GPT-4's API only accepts images as text
5. The image is **uploaded to Supabase Storage** with a random UUID filename (e.g., `a3f2504e.jpg`) so multiple users never collide
6. A **public URL** is constructed for the image — but there's a **bug here**: a second, different UUID is generated for the URL, so the URL doesn't actually point to the uploaded file

**The `seek(0)` calls** appear repeatedly because the uploaded file is a stream (like a tape). Every time you read it, the read-head moves forward. `seek(0)` rewinds it back to the start so the next read gets the full file again.

---

## Phase 2: Button Gate (Lines 67–81)

Two buttons appear side by side:
- **"Save to Favorites"** — saves the full plate image locally to `favorite_meals/`, then sets `run_script = True`
- **"Continue"** — just sets `run_script = True`

Everything below only runs if `run_script` is `True`. This prevents the expensive AI pipeline from firing on every Streamlit rerun (Streamlit reruns the entire script on every user interaction).

---

## Phase 3: GPT-4 Full-Plate Analysis (Lines 84–136)

A carefully crafted prompt asks GPT-4 Turbo (the **brain**) to analyze the food image and return **15 numbered points** in a specific format:

| # | Field | What it asks |
|---|-------|-------------|
| 1 | Description | What's on the plate (prose) |
| 2 | Items | List each food item |
| 3 | Calories | Per-item calorie estimates |
| 4 | Total Calories | Single aggregate number |
| 5 | Health Score | 0–10 (can be decimal like 6.5) |
| 6 | Rationale | Why that score |
| 7 | Macronutrient Estimate | Protein/fat/carbs in grams |
| 8 | Eat Frequency | "Can eat daily" / "Occasional treat" / "Avoid except rarely" |
| 9 | Comparison to Ideal Meal | How this stacks up vs a healthy benchmark |
| 10 | Mood/Energy Impact | Will it cause energy crash, etc. |
| 11 | Satiety Score | 0–10, how filling |
| 12 | Bloat Score | 0–10, bloating potential |
| 13 | Tasty Score | 0–10, how delicious it looks |
| 14 | Addiction Score | 0–10, craving-trigger potential |
| 15 | Summary | Brief closing with totals |

The image is sent embedded inside the API call as a base64 data URL (`data:image/jpeg;base64,...`).

**Parsing the response:** A regex splits GPT's text into 15 pieces by looking for the `N. **Label**: content` pattern. It:
- Finds each numbered bold header
- Captures everything after the colon
- Stops capturing when it hits the next number or the end of text

The 15 captured strings are assigned to named variables (`description_str`, `items_str`, etc.). **If GPT returns fewer than 15 items, this crashes with an IndexError** — there's no error handling.

---

## Phase 4: YOLOv8 Object Detection (Lines 138–149)

The uploaded image is converted from PIL format to OpenCV format (this involves an RGB-to-BGR color swap, because OpenCV historically uses Blue-Green-Red order).

**Two clean copies** of the image are saved (`img_cv2_orig` and `img_cv2_orig_c`). These are critical — the main `img_cv2` will get drawn on (boxes, overlays), but these copies stay pristine so that later, when cutting out food items, the cuts come from the unmodified original.

**YOLOv8m** (the **eyes**, medium model, auto-downloaded) scans the image and returns **bounding boxes** — rectangles around each detected object. Each box is four numbers: `[x1, y1, x2, y2]` (top-left corner and bottom-right corner in pixels).

---

## Phase 5: Per-Box Loop (Lines 151–304)

For **each bounding box** YOLO found, a multi-step sub-pipeline runs:

### Step 5a: Classify the crop (Lines 181–199)

- **Crop** the rectangle from the original image
- **Base64-encode** the crop
- **Send to GPT-4** (the **brain**) with a prompt that says: "Here's a close-up from a food plate. The full plate was described as: `{description_str}`. Which food item from that description matches this crop? Reply with just the name, or 'None'."
- GPT replies with a food name (e.g., `"rice"`) or `"None"`
- If `"None"` — this box is skipped entirely (maybe YOLO detected a plate rim or a fork)

### Step 5b: SAM segmentation (Lines 200–212)

- A **consistent random color** is assigned to each label (bright colors, 100–255 range, so they're always visible). Same label always gets the same color.
- **SAM is loaded** once (the **scissors** — ViT-H, the largest variant, ~636M parameters, moved to GPU). The full image embedding is computed once via `predictor.set_image()` — this is the expensive step, but it only happens once per image, not per box.
- For this box, SAM's `predict()` is called with the box coordinates as a prompt. **SAM returns 3 candidate masks** (because segmentation is ambiguous — e.g., "just the rice" vs "the rice plus the bowl" vs "everything in the box") along with a confidence score for each.
- The **highest-scoring mask** is selected — a boolean grid the size of the full image where `True` = "this pixel is the food item" and `False` = "background"
- The **complement mask** is computed (`logical_not`) — the exact opposite of the mask

### Step 5c: Extract two candidate images (Lines 214–232)

- **Segmented image**: Start with an all-black image. Copy pixels from the clean original wherever the mask says `True`. Result: the food item on a black background.
- **Complement image**: Same process with the flipped mask. Result: everything *except* the food item, on a black background.
- Both are **cropped to the bounding box** region (no need to send the full image to GPT)
- Both crops are converted to **grayscale** and pixels brighter than 30 are counted — this is the **brightness check**. Pixels darker than 30 are considered "black background" (the threshold accounts for JPEG compression artifacts and very dark foods). This count is used later to decide if the crop has enough real content to be worth saving.

### Step 5d: GPT-4 verification (Lines 234–261)

- Both cropped images (mask region = "A", complement = "B") are base64-encoded
- **GPT-4 (the brain) is called a third time** with both images and the prompt: "The expected food is `{label}`. Which segment — A or B — shows it? Reply A, B, or None."
- `max_tokens=5` — the answer should be a single character
- The response is cleaned (`.strip().upper()`)

**Why this step exists:** SAM (the scissors) sometimes cuts the wrong thing — it might outline the bowl instead of the food inside it. By showing GPT both the mask and its opposite, the code lets the brain pick the correct cut (or reject both).

### Step 5e: Save and overlay (Lines 262–304)

**File naming:** The code scans `food_library/` for existing files matching `{label}_N.jpg`, finds the highest N, and saves the new file as `{label}_{N+1}.jpg`. This prevents overwriting.

**Based on GPT's choice:**
- **"A"** — use the SAM mask; if the crop has >100 bright pixels, save the BGR-to-RGB-converted crop as a JPEG
- **"B"** — use the complement mask; same brightness check and save
- **"NONE"** — don't save anything, don't draw an overlay

**Mask overlay:** If a mask was selected, the code blends the random color onto the image at 60% opacity using the formula:
```
new_pixel = 0.6 * overlay_color + 0.4 * original_pixel
```
This makes the food item glow with a translucent color while the original texture shows through.

**Bounding box + label:** A colored rectangle is drawn around the box, and the food name is drawn 10 pixels above the top edge in the same color, with anti-aliased text.

---

## Phase 6: Display and Log (Lines 306–338)

After all boxes are processed:

1. The fully annotated image (`img_cv2`, now covered in colored overlays, boxes, and labels) is **converted BGR-to-RGB** for matplotlib
2. A matplotlib figure is created (10x8 inches, no axis ticks)
3. The figure **replaces** the original uploaded image in the same placeholder slot — the user sees a smooth transition from "plain photo" to "annotated analysis"
4. A **21-field dictionary** is built with all the GPT analysis results. Three fields (`macros_protein`, `macros_fat`, `macros_carb`) and `meal_time` are hardcoded to `"unspecified"` — parsing individual macro values from GPT's free-text response was never implemented
5. The record is **inserted into Supabase** via `supabase.table("food_logs").insert(record).execute()` — a single SQL INSERT behind the scenes
6. A green **"Logged in Supabase"** banner appears

---

## API Call Count Summary

For a plate where YOLO detects **N** objects and **K** of them are recognized as food (not "None"):

| Call | Count | Purpose |
|------|-------|---------|
| GPT-4 (full analysis) | 1 | 15-point nutritional analysis |
| GPT-4 (classify crop) | N | Match each YOLO box to a food name |
| SAM predict | K | Segment each recognized food item |
| GPT-4 (verify mask) | K | Confirm SAM picked the right region |
| **Total GPT-4 calls** | **1 + N + K** | |

For a typical plate with 5 detected objects and 4 recognized foods: **1 + 5 + 4 = 10 GPT-4 calls** per image.

---

## Known Issues

1. **Double UUID bug** (lines 58 vs 63): The image is uploaded with one UUID, but the URL is built with a different UUID — the URL is a 404
2. **Hardcoded SAM path** (line 171): Points to a Windows path `C:/Users/naren/...` — won't work on Linux
3. **Forced CUDA** (line 174): `sam.to("cuda")` crashes on CPU-only machines
4. **Fragile regex parsing** (line 121): If GPT deviates from the exact `N. **Label**: value` format, the app crashes with IndexError
5. **No error handling** on any API call — a network blip kills the whole app
6. **Function redefined inside loop** (line 239): `encode_cv2_to_base64` is recreated every iteration (wastes CPU, no functional harm)
7. **Macros never parsed** (lines 324–326): `macros_protein`, `macros_fat`, `macros_carb` are always `"unspecified"`

---

## Rebuilt Version (Flask)

The issues above are addressed in the rebuilt Flask version. The original Streamlit monolith is preserved at `reference/gpt_bhojan_app_v12.py`.

### Scaffolding Fixes

| Original Issue | Fix |
|---|---|
| Hardcoded SAM path | `config.py` uses relative `models/` path; `setup.sh` downloads checkpoint |
| Forced CUDA | `config.py:get_device()` auto-detects CUDA with allocation test, falls back to CPU |
| Fragile regex parsing | `analysis.py` uses GPT-4o JSON response mode — `json.loads()` instead of regex |
| No error handling | Flask route returns JSON errors; startup validates checkpoint + API key |
| Streamlit/Supabase coupling | Replaced with Flask + vanilla JS frontend; no external DB |
| SAM ViT-H (2.5 GB) | Switched to ViT-B (375 MB) — fits 8 GB VRAM alongside YOLO |

### Architectural Changes to the Pipeline

Beyond scaffolding, the rebuilt version makes several conceptual changes to how the pipeline works:

#### 1. Prompts extracted to `prompts.py`

All GPT-4o prompt text lives in `prompts.py` as named constants (`ANALYZE_PLATE_SYSTEM`, `CLASSIFY_AND_VALIDATE_CROP_SYSTEM`, `VERIFY_MASK_SYSTEM`). `analysis.py` imports them and passes them as system messages. This separates the *what GPT is told* from the *how the API is called*.

All three functions now use proper **system/user message separation** — the stable instructions (persona, rules, examples) go in the system message, and only the per-call variables (description, area ratio, images) go in the user message. This also enables OpenAI's prompt caching for the system message across calls.

#### 2. Area ratio is AUXILIARY, not a threshold

The original `classify_and_validate_crop` prompt told GPT a rule: "if the crop covers >40% of the image, it's probably the whole plate — reject it." This defeats the purpose of using an LLM. If the decision can be made by comparing a number to a threshold, you don't need a brain — you need an `if` statement.

The new `CLASSIFY_AND_VALIDATE_CROP_SYSTEM` prompt is a verbose, emphatic system prompt that:
- Frames the area ratio as ONE piece of context for a VISUAL judgment, not a decision criterion
- Uses ALL CAPS warnings against threshold-based reasoning
- Gives concrete visual examples of when large crops should be kept (a giant dosa) and small crops should be rejected (a fork handle)
- Instructs GPT to make the same judgment a human would when looking at the full plate and crop side-by-side

The only static threshold remaining is a **90% cost guard** — if a bounding box covers >90% of the image, it's clearly the whole frame and we skip it to save an API call. This is a cost optimization, not a quality filter.

#### 3. Two-pass pipeline with IoU dedup

The original version processed each box sequentially: classify → SAM → verify → overlay, one at a time. If YOLO produced two overlapping boxes that GPT both labeled "Sambar", you'd get duplicate segmentations, double-washed overlays, and wasted API calls.

The rebuilt version uses a two-pass structure:

**Pass 1 — Classification:** For every YOLO box, call GPT to classify it. Collect all (box, label) pairs.

**Dedup:** Group classified boxes by label. Within each label group, compute pairwise IoU (Intersection over Union). If two boxes with the same label have IoU > 0.5, keep the smaller one (tighter fit). This means two "Idli" boxes in different corners of the plate both survive, but two overlapping "Sambar" boxes collapse to one.

**Pass 2 — Segmentation + Verification:** Only surviving boxes proceed to SAM segmentation and GPT A/B verification.

#### 4. Mask complement bug fix

The original code had a critical bug in the A/B verification path. When GPT chose "B" (meaning SAM inverted its segmentation), the code applied `mask_c` — the complement of the SAM mask — as the overlay. But `mask_c` covers the **entire image** minus the food item, not just the inverted region within the bounding box. This caused the entire image to get color-washed when GPT said "B".

The fix constrains the inverted mask to the bounding box:
```python
# Old (broken): mask_final = mask_c
# New (fixed):
bbox_mask = np.zeros_like(mask)
bbox_mask[y1:y2, x1:x2] = True
mask_final = np.logical_and(mask_c, bbox_mask)
```

Now when GPT says "B", only the non-food pixels *inside that box* get the overlay.

### Revised API Call Count

For a plate where YOLO detects **N** objects, **K** pass GPT classification (not "None"), and **S** survive dedup:

| Call | Count | Purpose |
|------|-------|---------|
| GPT-4o (full analysis) | 1 | JSON-mode nutritional analysis |
| GPT-4o (classify crop) | N | Match each YOLO box to a food name |
| GPT-4o (verify mask) | S | Confirm SAM picked the right region |
| SAM predict | S | Segment each surviving food item |
| **Total GPT-4o calls** | **1 + N + S** | (S ≤ K ≤ N) |

The dedup step saves (K − S) GPT verify calls and (K − S) SAM segmentations compared to the original.

### Evaluation Results

A 17-image test suite (`eval/`) with ground truth labels yields:
- **Identification accuracy: 90%** (28/31 detections correct)
- **3 wrong IDs** — all the same pattern: GPT defaults to "Sambar" for unrecognized curry bowls
- **3 images with no output** (YOLO found no boxes) — not counted as failures
- Matching uses three rules: detected ⊂ GT, GT ⊂ detected, or reasonable semantic overlap (e.g., "Medu Vada" ↔ "Wada", "Potato stir-fry" ↔ "Subji")

Run the rebuilt version with `python app.py` (see `info/CLAUDE.md` for full setup instructions).
