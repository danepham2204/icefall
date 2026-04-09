# Introduction

Follow this: https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html

This CSTR VCTK Corpus includes speech data uttered by 110 English speakers with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive.
The newspaper texts were taken from Herald Glasgow, with permission from Herald & Times Group. Each speaker has a different set of the newspaper texts selected based a greedy algorithm that increases the contextual and phonetic coverage.
The details of the text selection algorithms are described in the following paper: [C. Veaux, J. Yamagishi and S. King, "The voice bank corpus: Design, collection and data analysis of a large regional accent speech database,"](https://doi.org/10.1109/ICSDA.2013.6709856).

The above information is from the [CSTR VCTK website](https://datashare.ed.ac.uk/handle/10283/3443).

# Data Preparation

Run `prepare.sh` to download and prepare the data. All stages are run by default.

**Option A — Download automatically (default):**
```bash
bash prepare.sh
```

**Option B — Use pre-existing local data (skip download):**

If you already have the VCTK corpus available locally (e.g. from [Kaggle](https://www.kaggle.com/datasets/pratt3000/vctk-corpus)
or another source), pass `--local-data-dir` to skip Stage 0 download:

```bash
bash prepare.sh --local-data-dir /path/to/your/VCTK
```

This will create a symlink at `download/VCTK` pointing to your local copy,
so all subsequent stages work without any modification.

# VITS

This recipe provides a VITS model trained on the VCTK dataset.

Pretrained model can be found [here](https://huggingface.co/zrjin/icefall-tts-vctk-vits-2024-03-18), note that this model was pretrained on the Edinburgh DataShare VCTK dataset.

For tutorial and more details, please refer to the [VITS documentation](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html).

The training command is given below:
```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./vits/train.py \
  --world-size 4 \
  --num-epochs 1000 \
  --start-epoch 1 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 350
```

To inference, use:
```
./vits/infer.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 500
```