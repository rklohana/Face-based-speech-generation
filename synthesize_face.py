"""Generate a WAV from new text and a face image with trained checkpoints.

Griffin-Lim reconstructs a waveform from the model's log-mel output. It is a
functional baseline; a vocoder trained on the exact mel configuration is
needed for high-quality speech.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from tools.extract_ecapa_centroids import EMBEDDING_DIM, unit_vector
from tools.extract_mels import MEL_CONFIG


WORD_OR_PUNCTUATION = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[,.!?;:]")


def read_lexicon(path: Path) -> dict[str, list[str]]:
    """Read a simple MFA pronunciation dictionary (first pronunciation wins)."""
    lexicon = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) < 2 or parts[0].startswith("#"):
                continue
            word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
            phones = parts[1:]
            # Some MFA dictionaries include one or more pronunciation probabilities.
            while phones:
                try:
                    float(phones[0])
                except ValueError:
                    break
                phones.pop(0)
            if phones and word not in lexicon:
                lexicon[word] = [phone.upper() for phone in phones]
    if not lexicon:
        raise ValueError(f"No pronunciations found in {path}")
    return lexicon


def phones_for_text(text: str, lexicon: dict[str, list[str]],
                    vocabulary: dict[str, int]) -> list[int]:
    tokens = WORD_OR_PUNCTUATION.findall(text)
    words = [token for token in tokens if token.isalpha() or "'" in token]
    if not words:
        raise ValueError("Text contains no words")
    phones = []
    for token in tokens:
        if token in ",.!?;:":
            if phones and phones[-1] != "sp":
                phones.append("sp")
            continue
        pronunciation = lexicon.get(token.lower())
        if pronunciation is None:
            raise ValueError(f"Word absent from MFA pronunciation dictionary: {token}")
        phones.extend(pronunciation)
    if phones and phones[-1] == "sp":
        phones.pop()
    missing = sorted(set(phones) - vocabulary.keys())
    if missing:
        raise ValueError(f"Pronunciation phones absent from TTS vocabulary: {missing}")
    return [vocabulary[phone] for phone in phones]


def invert_log_mel(log_mel, config: dict, iterations: int = 64):
    import torch
    import torchaudio

    if config != MEL_CONFIG:
        raise ValueError("Synthesis mel configuration differs from training")
    mel_magnitude = torch.exp(log_mel.clamp(-12, 8)).clamp_min(config["log_floor"])
    inverse = torchaudio.transforms.InverseMelScale(
        n_stft=config["n_fft"] // 2 + 1,
        n_mels=config["n_mels"], sample_rate=config["sample_rate"],
        f_min=config["f_min"], f_max=config["f_max"],
        norm=config["norm"], mel_scale=config["mel_scale"],
    ).to(log_mel.device)
    linear_magnitude = inverse(mel_magnitude).clamp_min(0)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=config["n_fft"], n_iter=iterations,
        win_length=config["win_length"], hop_length=config["hop_length"],
        power=config["power"],
    ).to(log_mel.device)
    waveform = griffin_lim(linear_magnitude)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("Vocoder returned an empty or nonfinite waveform")
    return waveform


def load_speaker_vector(args, expected_ecapa_config: dict):
    if args.speech_centroid:
        return unit_vector(np.load(args.speech_centroid, allow_pickle=False))
    import torch
    from PIL import Image

    from tools.face_encoder import image_transform, make_model

    checkpoint = torch.load(args.face_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("embedding_dim") != EMBEDDING_DIM or checkpoint.get("backbone") != "resnet18":
        raise ValueError("Face checkpoint has an incompatible embedding contract")
    if checkpoint.get("ecapa_config") != expected_ecapa_config:
        raise ValueError("Face and TTS checkpoints use different ECAPA speaker spaces")
    face = make_model(pretrained=False)
    face.load_state_dict(checkpoint["state_dict"])
    face = face.to(args.device).eval()
    with Image.open(args.face) as image:
        pixels = image_transform()(image.convert("RGB")).unsqueeze(0).to(args.device)
    with torch.no_grad():
        vector = face(pixels).squeeze(0).cpu().numpy()
    return unit_vector(vector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--face", type=Path)
    input_group.add_argument("--speech-centroid", type=Path,
                             help="Speech-conditioned baseline with the same TTS weights")
    parser.add_argument("--face-checkpoint", type=Path)
    parser.add_argument("--tts-checkpoint", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True,
                        help="Pronunciation dictionary used for MFA alignment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--griffin-lim-iterations", type=int, default=64)
    args = parser.parse_args()
    if args.face and not args.face_checkpoint:
        parser.error("--face needs --face-checkpoint")
    if args.griffin_lim_iterations < 1:
        parser.error("--griffin-lim-iterations must be positive")
    import torch
    from fastspeech2_model import FaceConditionedFastSpeech2

    checkpoint = torch.load(args.tts_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["mel_config"] != MEL_CONFIG or checkpoint["model_config"]["speaker_dim"] != EMBEDDING_DIM:
        raise ValueError("TTS checkpoint has an incompatible feature contract")
    vocabulary = checkpoint["vocabulary"]
    phone_ids = phones_for_text(args.text, read_lexicon(args.lexicon), vocabulary)
    vector = load_speaker_vector(args, checkpoint["ecapa_config"])
    model = FaceConditionedFastSpeech2(len(vocabulary), checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device).eval()
    phones = torch.tensor([phone_ids], dtype=torch.long, device=args.device)
    speaker = torch.from_numpy(vector).unsqueeze(0).to(args.device)
    with torch.no_grad():
        output = model(phones, speaker)
        mel = output["refined_mel"][0, :, :int(output["mel_lengths"][0])]
        waveform = invert_log_mel(mel, checkpoint["mel_config"], args.griffin_lim_iterations)
    samples = waveform.cpu().numpy()
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-8:
        raise ValueError("Synthesized waveform is silent")
    pcm = np.clip(samples / peak * 0.95, -1, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.output, MEL_CONFIG["sample_rate"], (pcm * 32767).astype(np.int16))
    print(f"Wrote {len(samples) / MEL_CONFIG['sample_rate']:.2f}s to {args.output}")


if __name__ == "__main__":
    main()
