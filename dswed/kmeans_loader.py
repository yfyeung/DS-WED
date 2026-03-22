from pathlib import Path

import joblib
from huggingface_hub import hf_hub_download

REPO_MAP = {
    "hubert-base": "yfyeung/kmeans-librispeech-hubert-base",
    "wavlm-base": "yfyeung/kmeans-librispeech-wavlm-base",
}


def download_kmeans_model(model_type, layer, vocab):
    """Download k-means model from HuggingFace if not cached."""
    if model_type not in REPO_MAP:
        raise ValueError(f"Unsupported model_type: {model_type}")

    repo_id = REPO_MAP[model_type]
    filename = f"{model_type}-layer-{layer}.kmeans.{vocab}.model"

    cache_dir = Path.home() / ".cache" / "dswed" / "kmeans"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir=str(cache_dir)
    )

    return joblib.load(model_path)
