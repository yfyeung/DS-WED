# DS-WED: Discretized Speech Weighted Edit Distance

<div align="center">
    <p>
    </p>
    <a href="https://arxiv.org/abs/2509.19928"><img src="https://img.shields.io/badge/arXiv-2509.19928-b31b1b.svg?logo=arXiv" alt="paper"></a>
    <a href="https://github.com/yfyeung/DS-WED"><img src="https://img.shields.io/badge/License-Apache%202.0-red.svg" alt="apache-2.0"></a>
    <p>
    Official code for <em>Measuring Prosody Diversity in Zero-Shot TTS: A New Metric, Benchmark, and Exploration</em>
    </p>
</div>

**DS-WED** is an objective diversity metric that quantifies prosodic variation via weighted edit distance over semantic tokens. DS-WED achieves substantially higher correlation with human judgments than existing acoustic metrics (e.g., log F0 RMSE and MCD), while remaining highly robust in speech tokenization from HuBERT and WavLM.

## Installation

```bash
git clone https://github.com/yfyeung/DS-WED.git
cd DS-WED
pip install -e .
```

## Quick Start

```python
import numpy as np
from dswed import DSWED

# Initialize the scorer
scorer = DSWED(
    model_type="hubert-base",
    vocab=50,
    layer=8,
)

# Compute distance between two audio samples
reference_wav = np.random.randn(16000)  # Reference audio (16kHz sampling rate)
generated_wav = np.random.randn(16000)  # Generated audio (16kHz sampling rate)

score = scorer.score(reference_wav, generated_wav)
print(f"DS-WED Score: {score}")
```

## Parameters

- `model_type`: Feature extraction model
  - `"hubert-base"`: HuBERT Base model
  - `"wavlm-base"`: WavLM Base model

- `vocab`: K-means clustering vocabulary size (20, 50, 100, 200, 500, 1000, or 2000)

- `layer`: Layer index for feature extraction (The index starts from 1, so if you want the 12-th layer feature, just set it to 12)

- `weights`: Edit distance weights (insertion, deletion, substitution), default (1, 1, 1)

## How It Works

1. Trim silence from audio using Silero VAD
2. Extract audio features using pre-trained self-supervised models (HuBERT or WavLM)
3. Quantize continuous features into discrete labels via K-means clustering
4. Map discrete labels to Unicode Chinese character sequences
5. Compute Levenshtein distance between the two sequences

## Citation

Please cite our paper if you find this work useful:

```bibtex
@inproceedings{yang2026dswed,
  title={Measuring Prosody Diversity in Zero-Shot TTS: A New Metric, Benchmark, and Exploration},
  author={Yifan Yang, Bing Han, Hui Wang, Long Zhou, Wei Wang, Mingyu Cui, Xu Tan, Xie Chen},
  booktitle={Proc. ICASSP},
  year={2026},
  address={Barcelona},
}
```
