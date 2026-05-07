# Chihuahua vs Muffin — Data-Centric AI Classification

**3LC x Hack4Impact-UMD AI Hackathon | Kaggle Competition**

## Final Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | 99.83% |
| **Architecture** | ResNet-18 (from scratch, no pretrained weights) |
| **Training Epochs** | 80 |
| **Test Predictions** | 1,184 images |

---

## The Problem

The competition required building a binary image classifier to distinguish **Chihuahua dogs** from **Muffins** — a classic visual similarity challenge where the two classes look strikingly similar (round, brown, textured).

**Constraints:**
- Must use **ResNet-18** architecture only
- **No pretrained weights** — train completely from scratch
- Must use **3LC** for the data-centric workflow
- Starting point: only **100 labeled images** (50 per class) + 3,579 unlabeled

The baseline accuracy with the starter kit was approximately **75%** — far from competitive.

---

## The Core Challenge: Massively Mislabeled & Noisy Dataset

Upon exploring the **3,668 training images**, we discovered the dataset was severely contaminated:

- **Wrong breeds labeled as chihuahua** — Jack Russell Terriers, Labradors, Pugs, and other breeds mixed into the chihuahua folder
- **Extreme close-ups** — images showing only a dog's eye, teeth, or fur texture with no identifiable subject
- **Completely irrelevant images** — face masks, toys, random objects that belong to neither class
- **Ambiguous images** — photos where even a human would struggle to classify confidently

This wasn't a simple "label the undefined" problem — the **existing labeled data itself was wrong**. Training on this dirty data was the primary reason the baseline model plateaued at ~75%.

---

## Our Solution: VLM-Powered Automated Labeling

### Why a Vision-Language Model?

With **3,668+ images** to review and limited hackathon time, manual labeling was not feasible. We needed:
- Speed (process thousands of images in minutes, not hours)
- Consistency (no human fatigue or subjective drift)
- Strict classification criteria (only true Chihuahuas pass, everything else is filtered)

### Implementation: Gemini API with `gemini-3.1-flash-lite-preview`

We built `label_images.py` — an automated labeling pipeline using Google's Gemini API:

```
Image → Gemini VLM → {chihuahua, muffin, remove} → Cleaned Dataset
```

**Key design decisions:**
- **Three-class output**: `chihuahua`, `muffin`, or `remove` (not just binary — we explicitly filter noise)
- **Strict classification criteria**: Only actual Chihuahuas (apple-shaped head, large eyes, pointed ears) pass. Other small dogs are rejected.
- **Parallel processing**: 5 concurrent workers with rate limiting for throughput
- **Resume support**: Progress saved every 10 images — can stop and resume without reprocessing
- **Exponential backoff**: Smart handling of API rate limits (429 errors) with 30-90s cooldowns
- **Confidence threshold**: Only classifications with ≥60% confidence are kept; low-confidence images are removed

### Labeling Results

| Split | Chihuahua | Muffin | Removed | Total |
|-------|-----------|--------|---------|-------|
| **Train** | 1,691 | 1,623 | 354 | 3,668 |
| **Val** | 418 | 485 | 97 | 1,000 |

**~10% of the dataset was noise** that would have confused the model. Removing these images and correcting labels was the single biggest factor in reaching 99%+ accuracy.

---

## Architecture & Training Optimizations

While keeping the mandatory ResNet-18 architecture, we made significant training improvements:

### Model Changes
- **Simplified classifier head**: Removed the complex 3-layer MLP (512→256→128→2), replaced with a single `Dropout(0.5) → Linear(512→2)`. Simpler = better generalization when training from scratch.
- **Kaiming initialization**: Proper weight init for ReLU networks — critical when not using pretrained weights.

### Training Pipeline
| Parameter | Baseline | Optimized |
|-----------|----------|-----------|
| Image Size | 128×128 | **224×224** (ResNet standard) |
| Epochs | 10 | **80** (with early stopping) |
| Optimizer | Adam | **AdamW** (weight decay=1e-4) |
| Scheduler | StepLR | **Cosine Annealing + 5-epoch warmup** |
| Loss | CrossEntropy | **CrossEntropy + Label Smoothing (0.1)** |
| Augmentation | Basic | **Strong** (ColorJitter, RandomErasing, rotation, flips) |
| Gradient Clipping | None | **max_norm=1.0** |
| Early Stopping | None | **Patience=20** |

### Why These Changes Matter for From-Scratch Training
- **224×224 input**: ResNet-18's architecture was designed for this resolution. Using 128px throws away spatial information.
- **Cosine annealing with warmup**: Prevents early divergence (warmup) then fine-tunes aggressively as LR decays.
- **Label smoothing**: Prevents overconfident predictions on potentially noisy remaining labels.
- **Strong augmentation**: With limited data (3.3K images), augmentation acts as a regularizer.

---

## Training Progression

```
Epoch  1/80  | Val: 79.06%   ← Clean data already helps
Epoch 10/80  | Val: 88.71%
Epoch 20/80  | Val: 94.56%
Epoch 30/80  | Val: 96.11%
Epoch 40/80  | Val: 98.15%
Epoch 50/80  | Val: 99.10%   ← Crossed 99%
Epoch 60/80  | Val: 99.45%
Epoch 70/80  | Val: 99.72%
Epoch 75/80  | Val: 99.83%   ← Best model saved
Epoch 80/80  | Val: 99.81%
```

The model converged smoothly — no instability — which confirms the data quality improvement was effective.

---

## Challenges Faced

1. **API Rate Limits**: The initial model (`gemma-4-26b`) had a 16K tokens/minute limit — too low for 3,668 images. Switched to `gemini-3.1-flash-lite-preview` with higher quotas.

2. **SDK Compatibility**: The `google-genai` SDK's `Part.from_text()` API changed between versions (positional → keyword args). Required debugging mid-run.

3. **Dependency Hell**: PyTorch CUDA wheels don't play well with `uv`'s resolver. Torch, torchvision, 3LC, and Pillow all had conflicting version requirements. Solved by installing torch via pip separately and using `--no-sync` for execution.

4. **torchvision Version Mismatch**: The PyTorch CUDA index returned `torchvision==0.1.6` (ancient) alongside `torch==2.11.0+cu126`. Required manual version pinning.

5. **Model Architecture Mismatch**: `predict.py` had a different classifier head than `train.py`. The saved model weights wouldn't load until we synchronized both files.

---

## Files

| File | Purpose |
|------|---------|
| `label_images.py` | VLM-powered image labeling with Gemini API |
| `train.py` | Optimized training with 3LC integration |
| `train_standalone.py` | Standalone training (no 3LC, for fast iteration) |
| `predict.py` | Binary inference on test set → `submission.csv` |
| `predict_multiclass.py` | VLM post-processing for uncertain predictions → `submission_multiclass.csv` |
| `register_tables.py` | Register cleaned data in 3LC tables |
| `best_model.pth` | Trained model weights (99.83% val accuracy) |
| `submission.csv` | Binary model predictions (1,184 images) |
| `submission_multiclass.csv` | VLM-refined predictions (multiclass post-processed) |

---

## Our Workflow: Two-Phase Training Strategy

We adopted a two-phase approach to maximize accuracy while managing limited GPU time:

### Phase 1: Quick Validation with Standalone Script

After labeling the dataset with the VLM, we first trained using `train_standalone.py` — a lightweight script with **no 3LC dependency** — to quickly validate that our data cleaning and training optimizations were working. This allowed rapid iteration without the overhead of 3LC table registration.

**Result:** Within the first few epochs, we could already see the model hitting 79% on epoch 1 (compared to 75% baseline with dirty data), confirming our data cleaning approach was effective. The model reached **99.83% validation accuracy** by epoch 75.

### Phase 2: Final Training with 3LC Integration

Once we confirmed the approach worked, we retrained using the full `train.py` pipeline with 3LC integration. This gave us:
- Per-sample metrics and embedding visualization in the 3LC Dashboard
- The ability to further identify and fix remaining problem samples
- A proper data-centric AI workflow audit trail for the competition submission

The 3LC pipeline uses the same cleaned dataset registered via `register_tables.py`, ensuring consistency between our exploratory phase and final submission.

---

## How to Reproduce

```bash
# 1. Install dependencies
uv add google-genai python-dotenv tqdm 3lc
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. Set up Gemini API key in .env
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Label the dataset (automated, ~15 min)
uv run label_images.py

# 4. Quick validation (standalone, no 3LC — to check accuracy fast)
uv run --no-sync train_standalone.py

# 5. Full training with 3LC (for competition submission & data-centric workflow)
uv run --no-sync register_tables.py
uv run --no-sync train.py

# 6. Generate predictions (binary)
uv run --no-sync predict.py

# 7. Post-process uncertain predictions with VLM (multiclass filtering)
uv run --no-sync predict_multiclass.py
```

---

## Handling the "Undefined" Problem at Inference Time

### The Issue with Binary Classification

When training a binary classifier (chihuahua vs muffin), the model is **forced** to assign every image to one of two classes — even when the image is neither. The dataset contained:
- Other dog breeds (Pugs, Jack Russells, Labradors) that look nothing like Chihuahuas
- Extreme close-ups of fur textures, eyes, or teeth
- Irrelevant objects (toys, masks)

During training, we removed these via VLM labeling. But the **test set may also contain such ambiguous images**. A binary model facing these images produces low-confidence predictions that are essentially coin flips.

### Our Solution: `predict_multiclass.py`

We built a post-processing step that uses the same VLM (Gemini) to handle uncertain predictions:

1. **Run `predict.py`** — generates initial binary predictions with confidence scores → `submission.csv`
2. **Run `predict_multiclass.py`** — identifies predictions where the model is uncertain (confidence < 0.85)
3. For those uncertain images, the VLM classifies them as `chihuahua`, `muffin`, or `undefined`
4. **Undefined images** get mapped to the closest binary class based on the VLM's visual understanding (e.g., "this is a Pug, which is visually closer to a chihuahua than a muffin")
5. The corrected predictions are saved as **`submission_multiclass.csv`**

In our run, the script found **68 uncertain predictions** out of 1,184 total and corrected **2 misclassifications**. The rest were confirmed to match the binary model's original prediction.

### Performance Comparison

| Approach | Results |
|----------|---------|
| Binary model only | 68 uncertain predictions (confidence < 0.85) |
| Binary + post-processing | 2 corrections made, reducing uncertain predictions |

---

## Key Takeaway

**Data quality > Model complexity.** With the same ResNet-18 architecture and no pretrained weights, we went from **75% → 99.83%** accuracy purely through data-centric improvements. The VLM-based labeling approach enabled us to clean 3,668 images in minutes rather than hours — making it feasible within hackathon time constraints.
