# LMSYS Chatbot Arena — Preference Prediction

Predict which LLM response users prefer (3-class: model A wins / model B wins / tie).

## Result

**No scored submission yet.** One submission was made (2026-04-07, the
Gemma2-2B QLoRA training notebook) and came back without a public score, so
nothing here has been measured against the leaderboard. The "Expected
Performance" section below is a projection, not a result.

## Quick Start

### 1. Download Competition Data
Download from https://www.kaggle.com/competitions/lmsys-chatbot-arena/data  
Extract into `data/` folder:
```
data/
  train.csv
  test.csv
  sample_submission.csv
```

### 2. Run EDA locally
Open `notebooks/01_eda.ipynb` and run all cells.

### 3. Train on Kaggle
1. Go to https://www.kaggle.com/ → New Notebook
2. Upload `notebooks/02_train.ipynb`
3. Add these **Input Datasets**:
   - `lmsys-chatbot-arena` (competition data)
   - `gemma-2-9b-it` (model weights — see below)
4. Set Accelerator → **GPU T4 ×2**
5. Run all cells
6. Save Version → **Save Output as New Dataset** (name it `lmsys-trained-adapter`)

### 4. Inference + Submit on Kaggle
1. Upload `notebooks/03_inference.ipynb`
2. Add these **Input Datasets**:
   - `lmsys-chatbot-arena`
   - `gemma-2-9b-it`
   - `lmsys-trained-adapter` (your training output)
3. **Turn OFF internet** (required for submission)
4. Run all cells → generates `submission.csv`
5. Submit

## Getting Gemma2-9B-IT Weights

1. Go to https://huggingface.co/google/gemma-2-9b-it
2. Accept the license agreement
3. Download the model files
4. Upload as a Kaggle Dataset named `gemma-2-9b-it`

Or search for existing uploads on Kaggle Datasets — many users have already uploaded `gemma-2-9b-it`.

## Project Structure
```
notebooks/
  01_eda.ipynb        — Exploratory data analysis (run locally)
  02_train.ipynb      — QLoRA fine-tuning (run on Kaggle GPU)
  03_inference.ipynb  — Inference + submission (run on Kaggle GPU)
data/                 — Competition data (not committed)
```

## Expected Performance

| Step | LB Score |
|------|----------|
| Public baseline (r=16, no TTA) | ~0.941 |
| + TTA | ~0.926 |
| + r=64, a=4, all modules | ~0.903 |
| + External 33k data | ~0.899 |
| + max_length 2048 + left truncation | ~0.895 |
