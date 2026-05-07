"""
Label images using Gemini API (gemma-4-26b-a4b-it) with parallel processing.

Strategy:
- Classify each image as "chihuahua", "muffin", or "remove"
- "remove" = other dog breeds, close-ups, irrelevant images, ambiguous content
- Move images to correct folders based on classification
- Process train/ (chihuahua, muffin, undefined) and val/ folders

Usage:
    uv run label_images.py
"""

import os
import sys
import json
import shutil
import base64
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Installing google-genai...")
    os.system("uv add google-genai python-dotenv")
    from google import genai
    from google.genai import types

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

MODEL_NAME = "gemini-3.1-flash-lite-preview"
MAX_WORKERS = 5  # Reduced to avoid rate limits
BASE_DIR = Path(__file__).parent / "data"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Output directories (cleaned)
CLEAN_TRAIN_DIR = BASE_DIR / "train_clean"
CLEAN_VAL_DIR = BASE_DIR / "val_clean"
REMOVED_DIR = BASE_DIR / "removed"

# ============================================================================
# GEMINI CLIENT
# ============================================================================

client = genai.Client(api_key=GEMINI_API_KEY)

# Resume support: track processed files
PROGRESS_FILE = Path(__file__).parent / "data" / "labeling_progress.json"


def load_progress() -> dict:
    """Load previously processed results."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    """Save processing progress for resume."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


CLASSIFICATION_PROMPT = """Classify this image as one of: chihuahua, muffin, or remove.

- chihuahua: A Chihuahua dog (small, apple-shaped head, large eyes, pointed ears).
- muffin: A muffin or cupcake (baked food).
- remove: Other dog breeds, close-ups (eye/teeth/fur), irrelevant objects, ambiguous.

Respond ONLY with JSON: {"label": "chihuahua" or "muffin" or "remove", "confidence": 0.0-1.0, "reason": "brief"}
"""


def encode_image(image_path: Path) -> str:
    """Read and base64 encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(image_path: Path) -> str:
    """Get MIME type from file extension."""
    ext = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/jpeg")


# Simple rate limiter
_rate_lock = threading.Lock()
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.5  # seconds between requests per thread


def classify_image(image_path: Path, retries: int = 5) -> dict:
    """Classify a single image using Gemini API with rate limiting."""
    global _last_request_time
    for attempt in range(retries):
        try:
            # Rate limiting
            with _rate_lock:
                now = time.time()
                wait = MIN_REQUEST_INTERVAL - (now - _last_request_time)
                if wait > 0:
                    time.sleep(wait)
                _last_request_time = time.time()

            image_data = encode_image(image_path)
            mime_type = get_mime_type(image_path)
            image_bytes = base64.b64decode(image_data)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type,
                            ),
                            types.Part(text=CLASSIFICATION_PROMPT),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=150,
                ),
            )

            text = response.text.strip()
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            result["file"] = str(image_path)
            return result

        except json.JSONDecodeError:
            text_lower = response.text.lower() if response and response.text else ""
            if "chihuahua" in text_lower and "muffin" not in text_lower:
                return {"label": "chihuahua", "confidence": 0.7, "reason": "parsed from text", "file": str(image_path)}
            elif "muffin" in text_lower and "chihuahua" not in text_lower:
                return {"label": "muffin", "confidence": 0.7, "reason": "parsed from text", "file": str(image_path)}
            else:
                return {"label": "remove", "confidence": 0.5, "reason": "ambiguous response", "file": str(image_path)}

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Rate limited — wait longer with exponential backoff
                wait_time = min(30 + (attempt * 15), 90)
                print(f"  Rate limited on {image_path.name}, cooling down {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"  Retry {attempt+1} for {image_path.name}: {e}")
                time.sleep(wait_time)
            else:
                print(f"  FAILED {image_path.name}: {e}")
                return {"label": "remove", "confidence": 0.0, "reason": f"error: {str(e)}", "file": str(image_path)}


def get_all_images(directory: Path) -> list:
    """Get all image files from a directory (recursive)."""
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    images = []
    if directory.exists():
        for f in directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in extensions:
                images.append(f)
    return images


def process_folder(folder_name: str, source_dir: Path, clean_dir: Path, progress: dict):
    """Process all images in a folder set (train or val) with resume support."""
    print(f"\n{'='*60}")
    print(f"  Processing: {folder_name}")
    print(f"{'='*60}")

    # Collect all images from all subfolders
    all_images = get_all_images(source_dir)
    if not all_images:
        print(f"  No images found in {source_dir}")
        return {}

    # Filter out already-processed images (resume support)
    already_done = set(progress.keys())
    pending_images = [img for img in all_images if str(img) not in already_done]
    skipped = len(all_images) - len(pending_images)

    print(f"  Found {len(all_images)} images total")
    if skipped > 0:
        print(f"  Resuming: {skipped} already processed, {len(pending_images)} remaining")

    # Create output directories
    (clean_dir / "chihuahua").mkdir(parents=True, exist_ok=True)
    (clean_dir / "muffin").mkdir(parents=True, exist_ok=True)
    (REMOVED_DIR / folder_name).mkdir(parents=True, exist_ok=True)

    # First, copy already-processed images to clean dirs
    results = {"chihuahua": [], "muffin": [], "remove": []}
    for img_path_str, info in progress.items():
        img_path = Path(img_path_str)
        if not img_path.exists():
            continue
        label = info.get("label", "remove")
        confidence = info.get("confidence", 0.0)
        if label in ("chihuahua", "muffin") and confidence >= 0.6:
            dest = clean_dir / label / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            results[label].append(img_path_str)
        else:
            dest = REMOVED_DIR / folder_name / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            results["remove"].append(img_path_str)

    if not pending_images:
        print(f"  All images already processed!")
        return results

    completed = 0
    save_interval = 10  # Save progress every N images

    # Process with parallel workers
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {
            executor.submit(classify_image, img_path): img_path
            for img_path in pending_images
        }

        for future in as_completed(future_to_path):
            img_path = future_to_path[future]
            try:
                result = future.result()
                label = result.get("label", "remove")
                confidence = result.get("confidence", 0.0)

                # Save to progress
                progress[str(img_path)] = {"label": label, "confidence": confidence, "reason": result.get("reason", "")}

                if label in ("chihuahua", "muffin") and confidence >= 0.6:
                    dest = clean_dir / label / img_path.name
                    if dest.exists():
                        stem = img_path.stem
                        dest = clean_dir / label / f"{stem}_{hash(str(img_path)) % 10000}{img_path.suffix}"
                    shutil.copy2(img_path, dest)
                    results[label].append(str(img_path))
                else:
                    dest = REMOVED_DIR / folder_name / img_path.name
                    if dest.exists():
                        stem = img_path.stem
                        dest = REMOVED_DIR / folder_name / f"{stem}_{hash(str(img_path)) % 10000}{img_path.suffix}"
                    shutil.copy2(img_path, dest)
                    results["remove"].append(str(img_path))

                completed += 1
                if completed % 20 == 0 or completed == len(pending_images):
                    print(f"  Progress: {completed}/{len(pending_images)} | "
                          f"chihuahua: {len(results['chihuahua'])} | "
                          f"muffin: {len(results['muffin'])} | "
                          f"removed: {len(results['remove'])}")

                # Periodically save progress
                if completed % save_interval == 0:
                    save_progress(progress)

            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                results["remove"].append(str(img_path))

    # Final save
    save_progress(progress)
    return results


def replace_original_with_clean(source_dir: Path, clean_dir: Path):
    """Replace original folder contents with cleaned data."""
    if not clean_dir.exists():
        return

    # Remove old labeled subfolders and replace with clean ones
    for class_name in ["chihuahua", "muffin"]:
        old_dir = source_dir / class_name
        new_dir = clean_dir / class_name

        if old_dir.exists():
            shutil.rmtree(old_dir)
        if new_dir.exists():
            shutil.copytree(new_dir, old_dir)
        else:
            old_dir.mkdir(parents=True, exist_ok=True)

    # Clear undefined folder (all labeled now)
    undefined_dir = source_dir / "undefined"
    if undefined_dir.exists():
        shutil.rmtree(undefined_dir)
        undefined_dir.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("  Chihuahua vs Muffin - Image Labeling with Gemini API")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Workers: {MAX_WORKERS}")
    print("=" * 60)

    # Load existing progress (resume support)
    progress = load_progress()
    print(f"  Loaded {len(progress)} previously processed results")

    all_results = {}

    # Process train set
    train_results = process_folder("train", TRAIN_DIR, CLEAN_TRAIN_DIR, progress)
    all_results["train"] = train_results

    # Process val set
    val_results = process_folder("val", VAL_DIR, CLEAN_VAL_DIR, progress)
    all_results["val"] = val_results

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for split, results in all_results.items():
        if results:
            print(f"\n  {split.upper()}:")
            print(f"    Chihuahua: {len(results.get('chihuahua', []))}")
            print(f"    Muffin:    {len(results.get('muffin', []))}")
            print(f"    Removed:   {len(results.get('remove', []))}")

    # Ask user before replacing
    print(f"\n{'='*60}")
    print("  Clean data is in:")
    print(f"    Train: {CLEAN_TRAIN_DIR}")
    print(f"    Val:   {CLEAN_VAL_DIR}")
    print(f"    Removed: {REMOVED_DIR}")
    print(f"{'='*60}")

    response = input("\nReplace original folders with cleaned data? (y/n): ").strip().lower()
    if response == "y":
        replace_original_with_clean(TRAIN_DIR, CLEAN_TRAIN_DIR)
        replace_original_with_clean(VAL_DIR, CLEAN_VAL_DIR)
        print("[OK] Original folders replaced with cleaned data.")
    else:
        print("[INFO] Clean data kept in separate folders. Move manually when ready.")

    # Save results log
    log_path = BASE_DIR / "labeling_results.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Results log saved to {log_path}")


if __name__ == "__main__":
    main()
