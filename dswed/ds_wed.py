import argparse
import logging
from argparse import Namespace

import torch
from Levenshtein import distance as levenshtein_distance
from silero_vad import get_speech_timestamps, load_silero_vad

from .kmeans_loader import download_kmeans_model
from .models import get_model


def int_array_to_chinese_unicode(arr):
    """
    Map each integer value in the array to a distinct Unicode Chinese character.
    Unicode region for Chinese characters: 4E00 - 9FFF (20992 characters)

    Args:
        arr (list): Array of integers.
    Returns:
        str: Unicode Chinese sentence.
    """
    # Base Unicode point for Chinese characters.
    base_unicode_point = 0x4E00

    # Convert each integer in the array to a Unicode Chinese character.
    unicode_sentence = "".join(chr(base_unicode_point + val) for val in arr)

    return unicode_sentence


class DSWED:
    def __init__(
        self,
        model_type="hubert-base",
        vocab=50,
        layer=-1,
    ):
        """
        Args:
            model_type (str): Model type. Select from "hubert-base (int): Number of vocabularies for k-means clustering. Select from 50, 100, 200.
            vocab (int): Number of vocabularies for k-means clustering. Select from 50, 100, 200.
            layer (int): Layer number to extract features. If None, the last layer is used.
        """
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        logging.basicConfig(format=formatter, level=logging.INFO)

        model_name, model_version = model_type.split("-")
        self.args = Namespace(
            model_name=model_name,
            model_version=model_version,
            layer_idx=layer,
            n_clusters=vocab,
        )
        logging.info(vars(self.args))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = get_model(self.args)
        self.model.eval()

        logging.info(
            f"Loading k-means model: {model_type}-layer-{layer}.kmeans.{vocab}"
        )
        self.km_model = download_kmeans_model(model_type, layer, vocab)
        self.vad_model = load_silero_vad()

    def get_parser(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--model-name",
            type=str,
            default="hubert",
            choices=["hubert", "wavlm"],
        )

        parser.add_argument(
            "--model-version",
            type=str,
            required=True,
        )

        parser.add_argument(
            "--layer-idx",
            type=int,
            default=8,
            help="The index starts from 1, so if you want the 12-th layer feature, just set it to 12",
        )

        parser.add_argument(
            "--n-clusters",
            type=int,
            default=50,
        )

        return parser.parse_args()

    def trim_silence(self, wav):
        timestamps = get_speech_timestamps(wav, self.vad_model)
        if len(timestamps) == 0:
            return wav
        start = timestamps[0]["start"]
        end = timestamps[-1]["end"]
        return wav[start:end]

    @torch.no_grad()
    def decode_label(self, audio):
        """
        Args:
            audio (torch.Tensor): Audio waveform tensor (1, T).
        Returns:
            list: List of integers.
        """
        audio = audio.to(self.device)
        batch = {"audio": audio, "audio_lens": torch.LongTensor([audio.shape[1]])}
        layer_results, embedding_lens = self.model.extract_features(
            batch, self.args.layer_idx
        )
        embedding = layer_results[0, : embedding_lens[0]]
        km_label = self.km_model.predict(embedding).tolist()
        return km_label

    def score(self, gt_wav, gen_wav, weights=(1, 1, 1)):
        """
        Args:
            gt_wav (np.ndarray): Ground truth waveform (T,).
            gen_wav (np.ndarray): Generated waveform (T,).
        Returns:
            float: Distance score.
        """
        gt_wav = torch.from_numpy(gt_wav).float()
        gen_wav = torch.from_numpy(gen_wav).float()

        gt_wav = self.trim_silence(gt_wav)
        gen_wav = self.trim_silence(gen_wav)

        gt_wav = gt_wav.unsqueeze(0).to(self.device)
        gen_wav = gen_wav.unsqueeze(0).to(self.device)

        gt_label = self.decode_label(gt_wav)
        gen_label = self.decode_label(gen_wav)
        gt_text = int_array_to_chinese_unicode(gt_label)
        gen_text = int_array_to_chinese_unicode(gen_label)
        dist_score = levenshtein_distance(gen_text, gt_text, weights=weights)
        return dist_score
