# 02_train.ipynb — Cell-by-Cell Explanation

> Both bugs (missing `PreferenceDataset` class and duplicate data cell) have been fixed.

---

## Cell 1 — Title & Instructions (Markdown)

Just a header. Lists what Kaggle datasets you need to add as "Input" before running the notebook.

---

## Cell 2 — Configuration (`class CFG`)

**What it does:** Defines every hyperparameter and path in one place so you only change things here.

**How it works:**
- `CFG` is a plain Python class used as a namespace (like a settings dictionary, but with dot access: `CFG.lora_r` instead of `config['lora_r']`).
- `model_size = '2b'` selects Gemma2-2B-IT. Change to `'9b'` for the larger model.
- After the class, an `if/else` sets `CFG.model_path` based on `model_size`.
- Path validation warns you if the model/data aren't attached on Kaggle.

**Key parameters explained:**

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `lora_r` | 64 | Rank of LoRA matrices — higher = more parameters = more capacity |
| `lora_alpha` | 128 | LoRA scaling factor. The effective scaling = `alpha / r` = 2.0. This multiplies every LoRA update. If too low (like 0.25), the model barely learns. |
| `lora_dropout` | 0.05 | Random dropout on LoRA layers for regularization |
| `lora_target_modules` | 7 modules | Which linear layers inside each transformer block get LoRA adapters |
| `max_length` | 512 | Max tokens per input. Longer = more context but slower & more memory |
| `per_device_batch_size` | 1 | Samples per GPU per forward pass. Limited by GPU memory (T4 = 16GB) |
| `gradient_accumulation_steps` | 8 | Simulates larger batches: does 8 forward passes, sums gradients, then updates. Effective batch = 8 × num_GPUs |
| `learning_rate` | 2e-5 | How big each weight update is. Standard for fine-tuning LLMs |
| `warmup_steps` | 100 | Linearly ramp up LR from 0 to `learning_rate` over 100 steps, prevents early instability |
| `weight_decay` | 0.01 | L2 regularization — pushes weights toward 0 to prevent overfitting |
| `use_4bit` | True | Load model weights in 4-bit (QLoRA) instead of 16-bit. Cuts memory ~4× |
| `bnb_4bit_quant_type` | 'nf4' | NormalFloat4 quantization — optimized for normally-distributed weights |
| `use_double_quant` | True | Quantize the quantization constants too. Saves ~0.4 bits/param extra |

**Read about:**
- [LoRA paper (Hu et al. 2021)](https://arxiv.org/abs/2106.09685) — the core idea of low-rank adaptation
- [QLoRA paper (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314) — 4-bit quantization + LoRA
- "Learning rate warmup" — why you start small and ramp up
- "Gradient accumulation" — how to simulate big batches on small GPUs

---

## Cell 3 — Imports & GPU Check

**What it does:** Imports all libraries and prints which GPUs are available.

**Key libraries:**

| Library | What it's for |
|---------|--------------|
| `torch` | PyTorch — the deep learning framework everything runs on |
| `transformers` | HuggingFace library — loads pretrained models, tokenizers, and provides the `Trainer` class |
| `peft` | Parameter-Efficient Fine-Tuning library — implements LoRA |
| `bitsandbytes` (installed in cell before) | Enables 4-bit quantization on GPU |
| `sklearn.metrics.log_loss` | Computes the competition metric (log loss) |

**Read about:**
- PyTorch basics: tensors, `.to(device)`, autograd
- What a tokenizer does (converts text → numbers)

---

## Cell 4 — Section Header (Markdown)

Just a section divider.

---

## Cell 5 — Load Training Data

**What it does:** Reads the competition CSV into a pandas DataFrame and optionally adds external data.

**How it works:**
1. `pd.read_csv()` loads the competition's `train.csv`
2. Each row has: `prompt` (user's question), `response_a` and `response_b` (two LLM answers), and three binary columns: `winner_model_a`, `winner_model_b`, `winner_tie` — exactly one is 1.0
3. If `external_data_path` is set, it loads extra training data and concatenates it
4. Prints class distribution (roughly 40% A wins, 40% B wins, 20% tie)

**Why no validation split:** In Kaggle competitions, every training row matters. We use the public leaderboard (LB) as our validation. This is common practice when data is limited.

**Read about:**
- pandas DataFrames basics
- Train/validation/test splits — and when to skip validation

---

## Cell 6 — ⚠️ DUPLICATE OF CELL 5

This is an accidental copy. **Delete this cell.**

---

## Cell 7 — Section Header (Markdown)

---

## Cell 8 — Load Model & Tokenizer

**What it does:** Loads Gemma2 in 4-bit quantized format with a classification head.

**How it works, step by step:**

### 1. Quantization config (`BitsAndBytesConfig`)
```
load_in_4bit=True       → store weights as 4-bit numbers (normally 16-bit)
bnb_4bit_quant_type='nf4'  → NormalFloat4 format (better than plain int4)
bnb_4bit_compute_dtype=float16  → do actual math in float16 (4-bit is storage only)
use_double_quant=True   → compress the quantization lookup tables too
```
This reduces a 9B model from ~18GB to ~5GB VRAM.

### 2. Tokenizer
- Converts text like `"Hello world"` → `[2, 4521, 2375]` (token IDs)
- `pad_token = eos_token`: Gemma doesn't have a pad token by default, so we reuse the end-of-sequence token
- `padding_side = 'right'`: Adds padding tokens on the right side of shorter sequences

### 3. Soft-capping removal
Gemma2 has "attention logit soft-capping" — it clips attention scores with `tanh` to prevent them from getting too large. But this operation is not supported by PyTorch's efficient SDPA (Scaled Dot Product Attention) kernel, which would force a slow fallback. Setting them to `None` **before** loading means the model uses the fast SDPA path. Accuracy impact is minimal.

### 4. Model loading
- `AutoModelForSequenceClassification` = base Gemma2 model + a new linear layer (`score`) on top that maps the last hidden state → 3 logits (one per class)
- `device_map='auto'` = automatically split model layers across available GPUs. On 2×T4, roughly half the layers go to each GPU.
- `torch_dtype=float16` = non-quantized layers (like the score head) use 16-bit floats
- `use_cache = False` = disables KV-cache (needed for training with gradient checkpointing; KV-cache is for inference speed only)

**Read about:**
- Model quantization: what it means to store weights in fewer bits
- Tokenization: BPE, SentencePiece, how text becomes numbers
- Attention mechanism: Q, K, V matrices, scaled dot-product
- SDPA vs eager attention in PyTorch
- `device_map='auto'` — how HuggingFace splits models across GPUs

---

## Cell 9 — LoRA Setup & Score Head Initialization

**What it does:** Wraps the quantized model with LoRA adapters and re-initializes the classification head.

**How it works, step by step:**

### 1. `prepare_model_for_kbit_training(model)`
- Sets all base model parameters to `requires_grad = False` (frozen)
- Enables gradient computation through quantized layers (normally quantized weights can't have gradients)
- Casts certain layers to float32 for stable training

### 2. Optional layer freezing
If `freeze_layers > 0`, the first N transformer layers are completely frozen (their LoRA adapters still get added but don't train). This can speed up training if you believe early layers are less task-specific.

### 3. `LoraConfig` and `get_peft_model`
LoRA works by adding small trainable matrices alongside the frozen weight matrices.

For each target module (e.g., `q_proj` with shape `[hidden_size, hidden_size]`):
- Adds matrix A with shape `[hidden_size, r]` (r=64)
- Adds matrix B with shape `[r, hidden_size]`
- The output becomes: `original_output + (input × A × B) × (alpha/r)`

So instead of training a `2304×2304` matrix (5.3M params), you train `2304×64 + 64×2304` = 295K params. Across all 7 target modules in all 26 layers, this gives ~83M trainable params out of 2.6B total (~3%).

The scaling factor `alpha/r = 128/64 = 2.0` controls how much influence the LoRA has. If this is too small (like 0.25 with the old alpha=16), updates barely affect the output and loss appears frozen.

`modules_to_save=['score']` tells PEFT to keep the classification head fully trainable (not LoRA'd), since it's newly initialized and needs full updates.

### 4. Preserving `hf_device_map`
PEFT wrapping creates new Python objects around the model. This can hide the `hf_device_map` attribute that tells HuggingFace Trainer "this model is already split across GPUs — don't try to wrap it in DataParallel." We manually copy it up to the outermost wrapper.

### 5. Score head re-initialization
The `score` layer is randomly initialized by HuggingFace with std=0.02, which can produce large logits → unstable loss. We re-initialize with a smaller std (`0.02 / sqrt(hidden_size)` ≈ 0.0004) so initial logits are near 0 → initial probabilities are near [0.33, 0.33, 0.33] → initial loss ≈ 1.1 (correct starting point for 3-class uniform prediction).

### 6. DataParallel fallback
If `hf_device_map` preservation didn't work, we set `model_parallel = True` as a backup signal to the Trainer.

**Read about:**
- [LoRA explained visually](https://lightning.ai/pages/community/lora-insights/) — how the low-rank matrices work
- Matrix decomposition / low-rank approximation — the math behind LoRA
- `requires_grad` in PyTorch — how you control which parameters get trained
- Weight initialization — why starting weights matter (Xavier, Kaiming, etc.)

---

## Cell 10 — Section Header (Markdown)

---

## Cell 11 — Custom Trainer (PreferenceTrainer)

**What it does:** Defines a custom trainer that handles soft labels and multi-GPU model parallelism.

**How it works:**

### `compute_loss` — Soft Cross-Entropy
Standard cross-entropy works with hard labels (class 0, 1, or 2). But our labels are soft/one-hot: `[1, 0, 0]` for "model A wins", `[0, 1, 0]` for "model B wins", `[0, 0, 1]` for "tie".

The formula: `loss = -Σ(labels × log(softmax(logits)))`

Step by step for one sample:
1. Model outputs 3 raw scores (logits), e.g. `[0.5, -0.2, 0.1]`
2. `softmax` converts to probabilities: `[0.40, 0.20, 0.27]` (sums to 1.0)
3. `log_softmax` = `log(softmax)`: `[-0.92, -1.61, -1.31]`
4. Multiply by labels `[1, 0, 0]` and sum: `-(-0.92) = 0.92`
5. Average across all samples in the batch

This is cast to `float32` even though the model is `float16` because `log` of small numbers can underflow in 16-bit.

### `_wrap_model` — Preventing DataParallel
When you have 2+ GPUs, HuggingFace Trainer normally wraps your model in `DataParallel` (which copies the model to each GPU and splits each batch). But our model is already split across GPUs via `device_map='auto'` (pipeline parallelism — different layers on different GPUs). These two approaches conflict and crash. By returning the model unchanged, we tell Trainer "I'll handle the GPUs myself."

**Read about:**
- Cross-entropy loss — the standard classification loss function
- Softmax function — converts raw scores to probabilities
- Log-likelihood — why we use log(probability)
- DataParallel vs Model Parallelism vs Pipeline Parallelism — different ways to use multiple GPUs
- [HuggingFace Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer)

---

## Cell 12 — ⚠️ MISSING: PreferenceDataset class

**This cell should exist but doesn't!** The `PreferenceDataset` class is used in cells 13 and 18 but was accidentally deleted during notebook reorganization.

It should define a PyTorch `Dataset` that:
1. Takes a DataFrame row
2. Formats the prompt + response_a + response_b into a single string with XML-like tags
3. Tokenizes it (with left truncation — if text is too long, cuts from the beginning since the end of responses is usually most informative)
4. Returns `input_ids`, `attention_mask`, and `labels` (the 3-class soft label vector)
5. Optionally applies label smoothing (blending hard labels toward uniform)

---

## Cell 13 — Create Dataset & Sanity Check

**What it does:** Creates the training dataset from the DataFrame and checks one sample.

**How it works:**
- Wraps the pandas DataFrame in a `PreferenceDataset` (PyTorch Dataset)
- The tokenizer converts each text sample into fixed-length token sequences (padded/truncated to `max_length=512`)
- Prints shape of `input_ids` (should be `[512]`) and the label vector
- Decodes first 50 tokens back to text to make sure the formatting looks right

**Read about:**
- PyTorch `Dataset` and `DataLoader` — how data is batched and fed to models
- Tokenizer padding and truncation

---

## Cell 14 — Section Header (Markdown)

---

## Cell 15 — Training Arguments & Training Loop

**What it does:** Configures the HuggingFace Trainer and runs training.

**Key arguments explained:**

| Argument | Value | Purpose |
|----------|-------|---------|
| `num_train_epochs` | 1 | One full pass through all training data (standard for LLM fine-tuning; more epochs risk overfitting) |
| `per_device_train_batch_size` | 1 | Process 1 sample per GPU per step (memory-limited) |
| `gradient_accumulation_steps` | 8 | Accumulate gradients over 8 mini-batches before updating. Effective batch = 8 |
| `learning_rate` | 2e-5 | Peak learning rate after warmup |
| `warmup_steps` | 100 | LR ramps from 0 → 2e-5 over first 100 steps |
| `lr_scheduler_type` | 'linear' | After warmup, LR decays linearly to 0 by end of training |
| `fp16` | True | Use mixed precision: forward pass in float16, gradients accumulated in float32. Faster + less memory |
| `max_grad_norm` | 1.0 | Gradient clipping — if the total gradient magnitude exceeds 1.0, scale it down. Prevents exploding gradients |
| `gradient_checkpointing` | True | Trade compute for memory: don't store intermediate activations, recompute them during backward pass. ~30% slower but ~50% less memory |
| `use_reentrant=False` | — | Newer, more reliable gradient checkpointing implementation |
| `save_strategy='epoch'` | — | Save a checkpoint after each epoch |
| `save_total_limit=1` | — | Keep only the latest checkpoint (saves disk space) |
| `remove_unused_columns=False` | — | Don't auto-remove columns from the dataset. Needed because our custom loss uses `labels` which Trainer would otherwise strip |

**`trainer.train()`** is where the actual training happens:
1. DataLoader creates batches from the dataset
2. For each batch: forward pass → compute loss → backward pass (compute gradients)
3. Every `gradient_accumulation_steps` batches: clip gradients → optimizer step → update LR
4. Logs loss every `logging_steps=10` optimizer steps

**Read about:**
- Gradient descent, backpropagation — how neural networks learn
- Learning rate schedules — warmup + decay
- Mixed precision training (fp16) — why it's faster
- Gradient clipping — preventing exploding gradients
- Gradient checkpointing — memory/speed tradeoff
- [HuggingFace TrainingArguments docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

---

## Cell 16 — Save LoRA Adapter

**What it does:** Saves the trained LoRA weights and tokenizer to disk.

**How it works:**
- `model.save_pretrained()` only saves the LoRA adapter weights (not the full 2.6B base model). This is typically ~200-400 MB instead of ~5 GB.
- Saves files: `adapter_config.json` (LoRA config), `adapter_model.safetensors` (trained weights), `tokenizer.json`, `tokenizer_config.json`, etc.
- During inference, you load the base model + merge these adapter weights back in.

**Read about:**
- SafeTensors format — safe, fast model weight storage
- LoRA adapter merging — how adapters get applied during inference

---

## Cell 17 — Section Header (Markdown)

---

## Cell 18 — Quick Validation

**What it does:** Runs the trained model on 100 training samples and computes log loss.

**How it works:**
1. `model.eval()` — switches off dropout and puts the model in inference mode
2. `torch.no_grad()` — disables gradient tracking (saves memory, faster)
3. For each sample: tokenize → send to GPU → get model logits → softmax → collect probabilities
4. `log_loss(targets, preds)` — computes the competition metric

**Why on training data:** This isn't a real validation — it's just checking that the model learned *something*. If the training loss went down but this log loss is still ~1.1, something is wrong. Real validation is on the Kaggle leaderboard.

**`next(model.parameters()).device`** — multi-GPU models have parameters on different GPUs. This gets the device of the first parameter (usually `cuda:0`), which is where inputs need to be sent. The model handles moving data between GPUs internally.

**Read about:**
- `model.eval()` vs `model.train()` — what changes (dropout, batch norm behavior)
- `torch.no_grad()` — why it's needed for inference
- Log loss (cross-entropy) — the competition evaluation metric
- Softmax — converting logits to probabilities

---

## Cell 19 — Next Steps (Markdown)

Instructions for saving the notebook output as a Kaggle Dataset so the inference notebook can load the adapter.

---

## Suggested Reading Order (Beginner Path)

1. **PyTorch basics** — tensors, GPU, autograd, `requires_grad`
2. **Neural network training loop** — forward pass, loss, backward pass, optimizer step
3. **Tokenization** — how text becomes numbers (BPE, SentencePiece)
4. **Transformer architecture** — attention, Q/K/V, feed-forward layers
5. **Transfer learning / fine-tuning** — why we start from a pretrained model
6. **Cross-entropy loss** — the standard classification objective
7. **LoRA paper** — the key technique this notebook uses
8. **QLoRA paper** — quantization + LoRA combined
9. **Mixed precision (fp16)** — faster training with lower precision
10. **HuggingFace Trainer** — the training loop abstraction we use

### Recommended Resources
- **3Blue1Brown "Neural Networks" playlist** — visual intuition for how NNs work
- **Andrej Karpathy "Let's build GPT"** (YouTube) — builds a transformer from scratch
- **HuggingFace NLP Course** (free) — covers tokenizers, models, Trainer
- **Sebastian Raschka "LLM from Scratch"** — book that builds everything step by step
- **LoRA paper**: https://arxiv.org/abs/2106.09685
- **QLoRA paper**: https://arxiv.org/abs/2305.14314
