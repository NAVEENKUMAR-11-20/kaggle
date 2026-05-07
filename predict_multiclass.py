"""
Multiclass prediction with undefined class handling.

Problem: In binary classification (chihuahua vs muffin), images that are neither
clearly a chihuahua nor a muffin (other dog breeds, close-up textures, irrelevant
objects) get forced into one of the two classes — often incorrectly.

Solution: This script uses the Gemini VLM as a post-processing filter to identify
images that the binary model is uncertain about, and reclassifies them using visual
understanding. Images identified as "undefined" (other breeds, textures, etc.) are
mapped to the most likely binary class based on visual similarity rather than the
model's uncertain prediction.

Workflow:
    1. Run predict.py first to get initial binary predictions with confidence scores.
    2. This script identifies low-confidence predictions (below threshold).
    3. For those uncertain images, it queries the Gemini VLM to determine if the image
       is truly a chihuahua, muffin, or something else (undefined).
    4. Undefined images are reassigned based on VLM's judgment of visual similarity.

Usage:
    python predict_multiclass.py

Outputs:
    submission.csv - Corrected submission with VLM-validated predictions.
"""

import csv
import base64
import json
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[ERROR] google-genai not installed. Run: uv add google-genai")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

SUBMISSION_PATH = Path("submission.csv")
TEST_DIR = Path("data/test")
OUTPUT_PATH = Path("submission_multiclass.csv")
CONFIDENCE_THRESHOLD = 0.85  # Re-evaluate predictions below this confidence
MODEL_NAME = "gemini-3.1-flash-lite-preview"
MAX_WORKERS = 5
MIN_REQUEST_INTERVAL = 0.5

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY not found in .env")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

# Rate limiting
_rate_lock = threading.Lock()
_last_request_time = 0.0

CLASSIFICATION_PROMPT = """Classify this image into one of three categories:

- chihuahua: A Chihuahua dog (small breed, apple-shaped head, large eyes, pointed ears)
- muffin: A muffin or cupcake (baked food item)
- undefined: Any other dog breed, close-up texture (fur/eye/teeth), irrelevant object, or ambiguous image

For undefined images, also state whether it looks MORE like a chihuahua or a muffin visually.

Respond ONLY with JSON: {"label": "chihuahua" or "muffin" or "undefined", "fallback": "chihuahua" or "muffin", "confidence": 0.0-1.0, "reason": "brief explanation"}
"""


def get_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    return mime_map.get(ext, "image/jpeg")


def classify_with_vlm(image_path: Path, retries: int = 3) -> dict:
    """Use VLM to classify an uncertain image into chihuahua/muffin/undefined."""
    global _last_request_time

    for attempt in range(retries):
        try:
            with _rate_lock:
                now = time.time()
                wait = MIN_REQUEST_INTERVAL - (now - _last_request_time)
                if wait > 0:
                    time.sleep(wait)
                _last_request_time = time.time()

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            mime_type = get_mime_type(image_path)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                            types.Part(text=CLASSIFICATION_PROMPT),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                ),
            )

            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(text)
            return result

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Error classifying {image_path.name}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)

    return {"label": "undefined", "fallback": "chihuahua", "confidence": 0.5}


def main():
    print("=" * 60)
    print("  Multiclass Post-Processing with VLM")
    print("=" * 60)

    if not SUBMISSION_PATH.exists():
        print(f"[ERROR] {SUBMISSION_PATH} not found. Run predict.py first.")
        return 1

    # Load current predictions
    rows = []
    with open(SUBMISSION_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"\n[1/3] Loaded {len(rows)} predictions from {SUBMISSION_PATH}")

    # Find low-confidence predictions
    uncertain = []
    for i, row in enumerate(rows):
        conf = float(row["confidence"])
        if conf < CONFIDENCE_THRESHOLD:
            uncertain.append((i, row))

    print(f"[2/3] Found {len(uncertain)} uncertain predictions (confidence < {CONFIDENCE_THRESHOLD})")

    if not uncertain:
        print("  No uncertain predictions to fix. Done.")
        return 0

    # Re-classify uncertain images with VLM
    print(f"\n[3/3] Re-classifying with VLM ({MODEL_NAME})...")
    fixed = 0
    for idx, (row_idx, row) in enumerate(uncertain):
        image_id = row["image_id"]
        image_path = TEST_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            # Try other extensions
            for ext in [".jpeg", ".png"]:
                alt = TEST_DIR / f"{image_id}{ext}"
                if alt.exists():
                    image_path = alt
                    break

        if not image_path.exists():
            continue

        result = classify_with_vlm(image_path)
        label = result.get("label", "undefined")

        # Map to binary prediction
        if label == "chihuahua":
            new_pred = "0"
        elif label == "muffin":
            new_pred = "1"
        else:
            # Undefined — use the VLM's fallback suggestion
            fallback = result.get("fallback", "chihuahua")
            new_pred = "0" if fallback == "chihuahua" else "1"

        if new_pred != row["prediction"]:
            rows[row_idx]["prediction"] = new_pred
            fixed += 1
            print(f"  [{idx+1}/{len(uncertain)}] {image_id}: {row['prediction']} -> {new_pred} ({label}, conf={result.get('confidence', 'N/A')})")
        else:
            print(f"  [{idx+1}/{len(uncertain)}] {image_id}: confirmed as {new_pred}")

    # Write corrected submission
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"  Fixed {fixed} predictions out of {len(uncertain)} uncertain images")
    print(f"  Saved to {OUTPUT_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
