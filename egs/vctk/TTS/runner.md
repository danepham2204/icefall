# VCTK VITS Training Runner (Kaggle)

> **Official docs:** https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html
> **Pretrained model (2024, use this one):** https://huggingface.co/zrjin/icefall-tts-vctk-vits-2024-03-18

> ⚠️ **Do NOT use the 2023 model** (`icefall-tts-vctk-vits-2023-12-05`).
> Its `tokens.txt` uses CMU Arpabet format (`<blk>`, `<sos/eos>`) which is incompatible
> with the current `tokenizer.py` that expects Piper IPA format (`_`, `^`, `$`).
> Using the 2023 model causes `KeyError: '_'` on startup.

---

## Quick Start: Run Inference with Pretrained Model (No Training Required)

If you just want to **generate speech audio** from an existing model, you do NOT need
to run `prepare.sh` or any data pipeline. You only need:

1. The icefall code (to run `infer.py`)
2. `epoch-1000.pt` — the trained model weights from HuggingFace
3. `data/tokens.txt` — the phoneme-to-ID map

### Cell 1 — Install Dependencies

```bash
# Install icefall repo
!git clone https://github.com/k2-fsa/icefall.git /kaggle/working/icefall
!pip install -r /kaggle/working/icefall/requirements.txt
!grep -v 'numba' /kaggle/working/icefall/requirements-tts.txt | pip install -r /dev/stdin
!pip install "numba>=0.59.0"

# Register icefall as Python package so "import icefall" works
!pip install -e /kaggle/working/icefall

# Install phonemizer used by VITS
!pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
```

### Cell 2 — Download the Pretrained Model

```python
from huggingface_hub import hf_hub_download
import os, shutil

MODEL_ID = "zrjin/icefall-tts-vctk-vits-2024-03-18"
BASE_DIR  = "/kaggle/working/icefall/egs/vctk/TTS"

os.makedirs(f"{BASE_DIR}/vits/exp", exist_ok=True)
os.makedirs(f"{BASE_DIR}/data", exist_ok=True)

# Download checkpoint (~1.08 GB).
# hf_hub_download preserves the repo path → saves to BASE_DIR/exp/epoch-1000.pt
# But infer.py / train.py expect it at BASE_DIR/vits/exp/epoch-1000.pt
# So: download first, then move to the correct location.
hf_hub_download(repo_id=MODEL_ID, filename="exp/epoch-1000.pt", local_dir=BASE_DIR)
shutil.copy2(f"{BASE_DIR}/exp/epoch-1000.pt", f"{BASE_DIR}/vits/exp/epoch-1000.pt")

# Phoneme token map (required by both train.py and infer.py)
hf_hub_download(repo_id=MODEL_ID, filename="data/tokens.txt", local_dir=BASE_DIR)

# Speaker ID list (required by infer.py to build speaker → int map)
hf_hub_download(repo_id=MODEL_ID, filename="data/speakers.txt", local_dir=BASE_DIR)

print("Ready.")
print(f"  Model  : {BASE_DIR}/vits/exp/epoch-1000.pt")
print(f"  Tokens : {BASE_DIR}/data/tokens.txt")
print(f"  Speakers: {BASE_DIR}/data/speakers.txt")
```

### Cell 3 — Run Inference

```bash
%cd /kaggle/working/icefall/egs/vctk/TTS

# Generate audio from the pretrained model
# Output is saved to: vits/exp/infer/epoch-1000/wav/
!CUDA_VISIBLE_DEVICES="0" python vits/infer.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 500
```

### Cell 4 — Play the Generated Audio

```python
import os
from IPython.display import Audio, display

wav_dir = "/kaggle/working/icefall/egs/vctk/TTS/vits/exp/infer/epoch-1000/wav"
wav_files = sorted(os.listdir(wav_dir))

# Play the first 3 generated audio files
for f in wav_files[:3]:
    print(f)
    display(Audio(os.path.join(wav_dir, f)))
```

> **That's it.** Steps 1–7 below are only needed if you want to continue training
> the model or understand the full pipeline.

---


## Core Concept

VITS (Variational Inference with adversarial learning for end-to-end TTS) is an
**end-to-end** text-to-speech model. Unlike traditional pipelines that chain acoustic
models with a separate vocoder, VITS takes text directly and outputs a waveform in one pass.

```
Text ──→ [Phoneme Encoder] ──→ [Latent Flow] ──→ [Waveform Decoder] ──→ Audio
```

**Training loop (simplified):**
1. Input text is converted to phoneme token IDs via `tokens.txt`
2. The encoder maps tokens to a latent representation
3. A normalizing flow samples a random variant of that latent
4. The HiFi-GAN decoder synthesizes a waveform from the latent
5. Two losses are computed: adversarial (discriminator) + reconstruction (mel-spectrogram)
6. Both the generator and discriminator are updated via backprop
7. Repeat for N epochs until the model converges (~1000 epochs)

**Key dependencies:**
| Library | Role |
|---|---|
| `k2` | FSA-based loss computation on GPU (must match CUDA + PyTorch version exactly) |
| `lhotse` | Audio dataset loading, manifest creation, spectrogram computation |
| `piper_phonemize` | Converts text → IPA phonemes used by VITS |
| `icefall` (package) | Training utilities, logging, data pipeline code |

---

## Environment (Kaggle)

- Python: 3.12
- PyTorch: 2.10.0+cu128
- CUDA: 12.8

---

## Step 1: Setup — Clone and Install Dependencies

```bash
# Clone the icefall repo
!git clone https://github.com/k2-fsa/icefall.git /kaggle/working/icefall

# Install base requirements
!pip install -r /kaggle/working/icefall/requirements.txt

# Install TTS requirements (skip numba 0.58.1 which is incompatible with Python 3.12)
!grep -v 'numba' /kaggle/working/icefall/requirements-tts.txt | pip install -r /dev/stdin
!pip install "numba>=0.59.0"

# Install lhotse (audio dataset toolkit)
!pip install lhotse

# Install k2 — must match CUDA 12.8 + PyTorch 2.10.0 exactly
# k2 is not on PyPI, use the custom index with -f
!pip install k2==1.24.4.dev20260306+cuda12.8.torch2.10.0 \
             -f https://k2-fsa.github.io/k2/cuda.html

# Install piper_phonemize — converts text to phonemes for VITS
!pip install piper_phonemize \
             -f https://k2-fsa.github.io/icefall/piper_phonemize.html

# Register the icefall repo as a Python package so "import icefall" works.
# Cloning alone does NOT make it importable — pip install -e registers it.
!pip install -e /kaggle/working/icefall
```

```text
PyTorch: 2.10.0+cu128
CUDA: 12.8
```

---

## Step 2: Verify Environment

```python
import torch
print("PyTorch:", torch.__version__)  # expected: 2.10.0+cu128
print("CUDA:", torch.version.cuda)    # expected: 12.8
import k2
print("k2:", k2.__version__)          # should import without error
```

---

## Step 3: Download Pretrained Checkpoint from HuggingFace

Instead of training from scratch (~2-3 days), download the pretrained `epoch-1000.pt`
and resume training from epoch 1001.

The HuggingFace repo contains:
```
icefall-tts-vctk-vits-2023-12-05/
├── data/
│   └── tokens.txt        ← phoneme-to-ID mapping (required for training)
└── exp/
    ├── epoch-1000.pt     ← model checkpoint (1.08 GB)
    ├── vits-epoch-1000.onnx
    └── vits-epoch-1000.int8.onnx
```

```python
from huggingface_hub import hf_hub_download
import os

MODEL_ID = "zrjin/icefall-tts-vctk-vits-2024-03-18"
BASE_DIR  = "/kaggle/working/icefall/egs/vctk/TTS"

os.makedirs(f"{BASE_DIR}/vits/exp", exist_ok=True)
os.makedirs(f"{BASE_DIR}/data", exist_ok=True)

# Download checkpoint (~1.08 GB) → vits/exp/epoch-1000.pt
hf_hub_download(repo_id=MODEL_ID, filename="exp/epoch-1000.pt",  local_dir=BASE_DIR)

# Download token file → data/tokens.txt
hf_hub_download(repo_id=MODEL_ID, filename="data/tokens.txt", local_dir=BASE_DIR)

print("Done.")
```

---

## Step 4: Prepare Data

> `train.py` does NOT read raw audio files directly.
> It reads `data/spectrogram/vctk_cuts_train.jsonl.gz` — a pre-computed manifest
> containing audio paths, durations, transcripts, and spectrograms.
> `prepare.sh` creates this file. **Missing this file → training crashes immediately.**

### Stage -1: Build the C extension for Monotonic Alignment Search

VITS uses monotonic alignment during training, implemented as a Cython C extension.
Build it once before any other stage.

```bash
%cd /kaggle/working/icefall/egs/vctk/TTS
!bash prepare.sh --stage -1 --stop_stage -1
```

```text
# Expected output:
Compiling core.pyx because it changed.
[1/1] Cythonizing core.pyx
...
copying build/lib.linux-x86_64-cpython-312/core.cpython-312-x86_64-linux-gnu.so ->
```

### Stage 1–4: Create Manifests, Spectrograms, Tokens, and Data Splits

The flag `--local-data-dir` does NOT work with parse_options.sh.
The correct approach (documented in prepare.sh itself) is to create a symlink manually:

```bash
# Point download/VCTK to your existing Kaggle dataset
!mkdir -p /kaggle/working/icefall/egs/vctk/TTS/download
!ln -sfv /kaggle/input/datasets/ \
          /kaggle/working/icefall/egs/vctk/TTS/download/VCTK

# Now run stages 1–4. Stage 0 (download) is automatically skipped
# because download/VCTK already exists.
%cd /kaggle/working/icefall/egs/vctk/TTS
!bash prepare.sh --stage 1 --stop_stage 4
```

What each stage does:

| Stage | What it creates |
|---|---|
| `1` | `data/manifests/vctk_*.jsonl.gz` — audio file list + transcripts |
| `2` | `data/spectrogram/vctk_cuts_all.jsonl.gz` — pre-computed spectrograms |
| `3` | Adds phoneme token IDs to each entry in the cuts file |
| `4` | Splits into `vctk_cuts_train.jsonl.gz` / `_valid` / `_test` |

---

## Step 5: Resume Training from Epoch 1001

`train.py` automatically looks for `vits/exp/epoch-<start_epoch - 1>.pt` and loads it.
With `--start-epoch 1001`, it loads `epoch-1000.pt` (downloaded from HuggingFace in Step 3).

```bash
%cd /kaggle/working/icefall/egs/vctk/TTS

!CUDA_VISIBLE_DEVICES="0" python vits/train.py \
  --world-size 1 \
  --num-epochs 1100 \
  --start-epoch 1001 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 350
```

**Arguments explained:**

| Argument | Value | Meaning |
|---|---|---|
| `--world-size` | `1` | Number of GPUs. `1` = single GPU (no distributed training) |
| `--num-epochs` | `1100` | Train until epoch 1100 total |
| `--start-epoch` | `1001` | Resume from epoch 1001; loads `epoch-1000.pt` automatically |
| `--exp-dir` | `vits/exp` | Output directory for checkpoints, logs, and TensorBoard |
| `--tokens` | `data/tokens.txt` | Maps each phoneme character to an integer ID for the model |
| `--max-duration` | `350` | Max total audio seconds per batch. Reduce to 200 if GPU runs OOM |

Checkpoints are saved to `vits/exp/epoch-1001.pt`, `epoch-1002.pt`, etc.

---

## Step 6: Inference

Run inference to check audio quality at any checkpoint.

```bash
!CUDA_VISIBLE_DEVICES="0" python vits/infer.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 500
```

Output audio is saved to: `vits/exp/infer/epoch-1000/wav/`

---

## Step 7: Export to ONNX (for deployment)

After training, export the model to ONNX for inference without PyTorch.

```bash
!python vits/export-onnx.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt
```

This generates:
- `vits/exp/vits-epoch-1000.onnx` — full precision (126 MB)
- `vits/exp/vits-epoch-1000.int8.onnx` — quantized int8 (47 MB, faster inference)

---

## Full Pipeline Summary

```
[Step 1] Install: icefall + k2 + piper_phonemize + lhotse
         ↓
[Step 2] Verify: PyTorch 2.10.0, CUDA 12.8, k2 imports OK
         ↓
[Step 3] Download pretrained checkpoint from HuggingFace
         → vits/exp/epoch-1000.pt  (1.08 GB)
         → data/tokens.txt
         ↓
[Step 4] prepare.sh
         Stage -1 → build monotonic_align C extension
         Stage 1  → create audio manifests (lhotse)
         Stage 2  → compute spectrograms
         Stage 3  → add phoneme tokens to cuts
         Stage 4  → split into train / valid / test
         → data/spectrogram/vctk_cuts_train.jsonl.gz
         ↓
[Step 5] train.py --start-epoch 1001
         → loads epoch-1000.pt, continues training
         → saves epoch-1001.pt, epoch-1002.pt, ...
         ↓
[Step 6] infer.py
         → generates audio samples to check quality
         ↓
[Step 7] export-onnx.py
         → vits-epoch-1000.onnx  (deploy anywhere)
```
