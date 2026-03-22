import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor


def get_model(args):
    model_name = args.model_name
    if model_name == "wavlm":
        model = WavLMModel(model_version=args.model_version)
    elif model_name == "hubert":
        model = HuBERTModel(model_version=args.model_version)
    else:
        raise ValueError(f"{model_name} is not supported yet")

    return model


class HuBERTModel(torch.nn.Module):
    def __init__(self, model_version: str = "base"):
        super().__init__()
        if model_version == "large":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                f"facebook/hubert-{model_version}-ll60k"
            )
            self.model = AutoModel.from_pretrained(
                f"facebook/hubert-{model_version}-ll60k"
            )
        elif model_version == "base":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                f"facebook/hubert-{model_version}-ls960"
            )
            self.model = AutoModel.from_pretrained(
                f"facebook/hubert-{model_version}-ls960"
            )
            self.processor.do_normalize = False
        else:
            raise ValueError(f"Unseen model version: {model_version}")

    def extract_features(self, batch, layer_idx, return_numpy=True):
        audio_pt = batch["audio"]
        if isinstance(audio_pt, list):
            audios = [audio.numpy() for audio in audio_pt]
        else:
            audio_lens_pt = batch["audio_lens"]
            audios = []
            for i in range(audio_pt.shape[0]):
                audios.append(audio_pt[i, : audio_lens_pt[i]].tolist())

        device = next(self.model.parameters()).device
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.model(output_hidden_states=True, **inputs)
        layer_results = outputs.hidden_states[layer_idx]
        padding_mask = self.model._get_feature_vector_attention_mask(
            layer_results.shape[1], inputs["attention_mask"]
        )
        embedding_lens = padding_mask.sum(dim=1)

        if return_numpy:
            layer_results = layer_results.cpu().numpy()
        return layer_results, embedding_lens


class WavLMModel(torch.nn.Module):
    def __init__(self, model_version: str = "base"):
        super().__init__()
        from transformers import WavLMModel as HFWavLM

        model_name = f"microsoft/wavlm-{model_version}"
        self.model = HFWavLM.from_pretrained(model_name)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def extract_features(self, batch, layer_idx):
        audio_pt = batch["audio"]
        if isinstance(audio_pt, list):
            audios = [audio.numpy() for audio in audio_pt]
        else:
            audio_lens_pt = batch["audio_lens"]
            audios = []
            for i in range(audio_pt.shape[0]):
                audios.append(audio_pt[i, : audio_lens_pt[i]].tolist())

        device = next(self.model.parameters()).device
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.model(output_hidden_states=True, **inputs)
        layer_results = outputs.hidden_states[layer_idx]
        padding_mask = self.model._get_feature_vector_attention_mask(
            layer_results.shape[1], inputs["attention_mask"]
        )
        embedding_lens = padding_mask.sum(dim=1)

        return layer_results.cpu().numpy(), embedding_lens
