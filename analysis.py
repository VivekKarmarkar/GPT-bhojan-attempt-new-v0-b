import json
from openai import OpenAI
from config import OPENAI_API_KEY
from prompts import (
    ANALYZE_PLATE_SYSTEM,
    CLASSIFY_AND_VALIDATE_CROP_SYSTEM,
    VERIFY_MASK_SYSTEM,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def analyze_image(b64_image: str) -> dict:
    """Full-plate analysis via GPT-4o with JSON response mode.

    Returns dict with keys: description, items, health, satiety, bloat, tasty, addiction.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYZE_PLATE_SYSTEM},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]
        }],
        max_tokens=300,
    )
    return json.loads(response.choices[0].message.content)


def classify_and_validate_crop(b64_full: str, b64_crop: str, area_ratio: float, description: str) -> str:
    """Classify a YOLO crop AND validate it as an individual food item.

    GPT-4o sees the full plate image, the cropped bounding box, the area ratio,
    and the plate description. It makes a VISUAL judgment about whether the crop
    isolates a single food item.

    Returns food item name or 'None'.
    """
    user_text = (
        f"The plate was described as: {description}\n"
        f"The crop covers {area_ratio:.0%} of the full image area.\n\n"
        "Image 1 is the full plate. Image 2 is the cropped bounding box."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CLASSIFY_AND_VALIDATE_CROP_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_full}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_crop}"}}
                ]
            }
        ],
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()


def verify_mask(label: str, img_a_b64: str, img_b_b64: str) -> str:
    """A/B verification — which segment matches the expected food item?

    Returns 'A', 'B', or 'NONE'.
    """
    user_text = f"The expected food item is: '{label}'."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": VERIFY_MASK_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b_b64}"}}
                ]
            }
        ],
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().upper()
