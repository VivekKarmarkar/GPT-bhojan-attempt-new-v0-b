# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 06:58:55 2025

@author: vivek
"""

import base64
from datetime import datetime
import os
import re
import uuid
import io
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

import streamlit as st
from openai import OpenAI
from segment_anything import sam_model_registry, SamPredictor
from supabase import create_client, Client

# === Streamlit Config ===
st.set_page_config(page_title="GPT Bhojan 🍛", layout="centered")

# === API + Supabase setup ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# === Title and file uploader ===
st.title("🍛 GPT Bhojan")
st.write("Upload a food photo and get insights: description, calories, health score, and more!")

uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

# === Ensure output directory exists ===
os.makedirs("food_library", exist_ok=True)
os.makedirs("favorite_meals", exist_ok=True)

if uploaded_file is not None:
    timestamp = datetime.now().isoformat()

    # Create placeholder for replacing image later
    image_placeholder = st.empty()

    # Show original uploaded image
    uploaded_file.seek(0)
    image_placeholder.image(uploaded_file, caption="Your uploaded food plate", use_column_width=True)
    
    # Prepare image for Supabase + GPT
    uploaded_file.seek(0)
    base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")
    uploaded_file.seek(0)
    supabase.storage.from_("foodimages").upload(
        f"{uuid.uuid4()}.jpg",
        uploaded_file.getvalue(),
        {"content-type": "image/jpeg"}
    )

    image_name = f"{uuid.uuid4()}.jpg"
    image_url = f"{st.secrets['SUPABASE_URL'].replace('.supabase.co', '.supabase.co/storage/v1/object/public')}/foodimages/{image_name}"
    
    # Run script flag
    run_script = False
        
    col1, col2 = st.columns(2)

    with col1:
        if st.button("❤️ Save this full plate to Favorites"):
            uploaded_file.seek(0)
            meal_image = Image.open(uploaded_file).convert("RGB")
            meal_image.save(f"favorite_meals/{image_name}")
            st.success("✅ Meal saved to your favorites!")
            run_script = True
    
    with col2:
        if st.button("🚆 Continue..."):
            run_script = True

    if run_script:
        # === GPT Prompt ===
        prompt = (
            "You are GPT Bhojan 🍛, a food and nutrition assistant.\n\n"
            "Please analyze the food in this image and return a structured analysis in this format:\n\n"
            "1. **Description**: A short paragraph describing the food.\n"
            "2. **Items**: A list of distinct items on the plate.\n"
            "3. **Calories**: Estimate calories for each item and the total.\n"
            "4. **Total Calories**: Tell me the total calorie estimate.\n"
            "5. **Health Score**: Give a score from 0 to 10 (real number).\n"
            "6. **Rationale**: Explain why this score was given.\n"
            "7. **Macronutrient Estimate**: Rough protein (g), fat (g), carbs (g).\n"
            "8. **Eat Frequency**: Label as one of ['Can eat daily', 'Occasional treat', 'Avoid except rarely'].\n"
            "9. **Comparison to Ideal Meal**: Brief comment on how this compares to a typical healthy benchmark meal.\n"
            "10. **Mood/Energy Impact**: What short-term effects might this food have (e.g., energy crash, satiety)?\n"
            "11. **Satiety Score**: Score from 0 to 10 based on how full this meal is likely to make the person feel.\n"
            "12. **Bloat Score**: Score from 0 to 10 based on how much bloating this meal might cause.\n"
            "13. **Tasty Score**: Score from 0 to 10 based on how tasty this meal is likely to be (based on visual and content).\n"
            "14. **Addiction Score**: Score from 0 to 10 based on how likely this meal is to trigger addictive eating patterns.\n"
            "15. **Summary**: Total calorie estimate with final health score and brief closing note."
        )
    
        with st.spinner("Analyzing your plate..."):
            # === GPT-4 Analysis ===
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]
            )
            result = response.choices[0].message.content
            st.markdown("### 🧠 Analysis:")
            st.write(result)
    
            # === Parse fields ===
            matches = re.findall(r'\d+\.\s\*\*.*?\*\*:\s*(.*?)(?=\n\d+\.|\Z)', result, re.DOTALL)
            description_str = matches[0]
            items_str = matches[1]
            calories_str = matches[2]
            total_calories_str = matches[3]
            health_score_str = matches[4]
            rationale_str = matches[5]
            macronutrient_estimate_str = matches[6]
            eat_frequency_str = matches[7]
            ideal_comparison_str = matches[8]
            mood_impact_str = matches[9]
            satiety_score_str = matches[10]
            bloat_score_str = matches[11]
            tasty_score_str = matches[12]
            addiction_score_str = matches[13]
            summary_str = matches[14]
    
            # === YOLO Setup ===
            uploaded_file.seek(0)
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # === Create copy of original (unmodified) image ===
            img_cv2_orig = img_cv2.copy()
            img_cv2_orig_c = img_cv2.copy()
    
            model = YOLO("yolov8m.pt")
            results = model(img_cv2)
            boxes = results[0].boxes.xyxy
    
            prompt_template = (
                "You are given a close-up image of one item from a larger food plate, along with a textual description "
                "of the full plate. Based only on the food names mentioned in the description, identify whether the image "
                "contains any of the described food items.\n\n"
                "If there's a match, respond with just the food name. If unsure or no match, respond with 'None'.\n\n"
                f"Description of full plate: {description_str}"
            )
    
            color_map = {}
            def get_color(label):
                if label not in color_map:
                    color_map[label] = tuple(random.randint(100, 255) for _ in range(3))
                return color_map[label]
    
            def image_to_base64(image: Image.Image):
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # === SAM setup ===
            sam_checkpoint = "C:/Users/naren/segment-anything/models/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to("cuda")  # or "cpu"
            predictor = SamPredictor(sam)
    
            # Set the SAM image (RGB NumPy array)
            predictor.set_image(np.array(pil_img))  # PIL image already loaded earlier
                    
            # === Box-wise GPT classification + SAM segmentation ===
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.tolist())
                bb_img = pil_img.crop((x1, y1, x2, y2))
                img_b64 = image_to_base64(bb_img)
    
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_template},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }],
                    max_tokens=50,
                )
                label = response.choices[0].message.content.strip()
    
                if label.lower() != "none":
                    color = get_color(label)
                    
                    # === SAM with box prompt instead of center-point ===
                    input_box = np.array([[x1, y1, x2, y2]])  # box format: [x1, y1, x2, y2]
                    
                    masks, scores, _ = predictor.predict(
                        box=input_box,
                        multimask_output=True
                    )
    
                    best_idx = np.argmax(scores)
                    mask = masks[best_idx]
                    mask_c = np.logical_not(mask)
                    
                    # === Extract segmented region from original (unmodified) image ===
                    segmented = np.zeros_like(img_cv2_orig)
                    segmented[mask] = img_cv2_orig[mask]
                    
                    # === Extract complement of segmented region from original (unmodified) image ===
                    segmented_c = np.zeros_like(img_cv2_orig_c)
                    segmented_c[mask_c] = img_cv2_orig_c[mask_c]
                    
                    # === Crop the bounding box region from the segmented image ===
                    segmented_crop = segmented[y1:y2, x1:x2]
                    segmented_crop_c = segmented_c[y1:y2, x1:x2]
                    
                    # Convert to grayscale
                    gray_crop = cv2.cvtColor(segmented_crop, cv2.COLOR_BGR2GRAY)
                    gray_crop_c = cv2.cvtColor(segmented_crop_c, cv2.COLOR_BGR2GRAY)
                    
                    # Create a binary mask of "non-black" pixels
                    bright_pixels = gray_crop > 30  # tweak threshold if needed
                    bright_pixels_c = gray_crop_c > 30
                    
                    # Assign labels to crop for GPT sanity check
                    crop_a = segmented_crop
                    crop_b = segmented_crop_c
                    
                    # Convert to PIL and base64
                    def encode_cv2_to_base64(img_arr):
                        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        buffered = io.BytesIO()
                        pil_img.save(buffered, format="PNG")
                        return base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    img_a_b64 = encode_cv2_to_base64(crop_a)
                    img_b_b64 = encode_cv2_to_base64(crop_b)
                    
                    # GPT Verification Prompt
                    gpt_mask_check = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"The expected food item is: '{label}'.\n\nYou are shown two segments (A and B) extracted from a food photo. Exactly one may correspond to the expected food item, or possibly neither. Reply with just one of:\n'A', 'B', or 'None'."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a_b64}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b_b64}"}}
                            ]
                        }],
                        max_tokens=5,
                    )
                    
                    gpt_choice = gpt_mask_check.choices[0].message.content.strip().upper()
                    
                    # Directory and label
                    base_path = "food_library"
                    pattern = re.compile(rf"^{re.escape(label)}_(\d+)\.jpg$")
                    
                    # List all matching files and extract numeric suffixes
                    existing_nums = []
                    for fname in os.listdir(base_path):
                        match = pattern.match(fname)
                        if match:
                            existing_nums.append(int(match.group(1)))
                    
                    # Determine next available number
                    next_n = max(existing_nums) + 1 if existing_nums else 1
                    filename = os.path.join(base_path, f"{label}_{next_n}.jpg")
                    
                    mask_final = None
                    if gpt_choice == "A":
                        mask_final = mask
                        if np.sum(bright_pixels) > 100:
                            segmented_crop_rgb = cv2.cvtColor(segmented_crop, cv2.COLOR_BGR2RGB)
                            Image.fromarray(segmented_crop_rgb).save(filename)
                            
                    elif gpt_choice == "B":
                        mask_final = mask_c
                        if np.sum(bright_pixels_c) > 100:
                            segmented_crop_rgb_c = cv2.cvtColor(segmented_crop_c, cv2.COLOR_BGR2RGB)
                            Image.fromarray(segmented_crop_rgb_c).save(filename)
                        
                    # === Blend SAM mask on image with same color (transparency) ===
                    if mask_final is not None:
                        overlay_color = np.array(color).astype(np.uint8)
                        alpha = 0.6  # transparency
        
                        img_cv2[mask_final] = (
                            alpha * overlay_color + (1 - alpha) * img_cv2[mask_final]
                        ).astype(np.uint8)
    
                    # === Draw BB + Label as before ===
                    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=color, thickness=3)
                    cv2.putText(img_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
            # === Display the final image ===
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img_rgb)
            ax.axis("off")
            image_placeholder.pyplot(fig)
            
            # === Log to Supabase ===
            record = {
                "timestamp": timestamp,
                "meal_time": "unspecified",
                "description": description_str,
                "items": items_str,
                "calories": calories_str,
                "total_calories": total_calories_str,
                "health_score": health_score_str,
                "rationale": rationale_str,
                "macronutrient_estimate": macronutrient_estimate_str,
                "macros_protein": "unspecified",
                "macros_fat": "unspecified",
                "macros_carb": "unspecified",
                "eat_frequency": eat_frequency_str,
                "ideal_comparison": ideal_comparison_str,
                "mood_impact": mood_impact_str,
                "satiety_score": satiety_score_str,
                "bloat_score": bloat_score_str,
                "tasty_score": tasty_score_str,
                "addiction_score": addiction_score_str,
                "summary": summary_str,
                "image_url": image_url
            }
            supabase.table("food_logs").insert(record).execute()
            st.success("Logged in Supabase ✅")