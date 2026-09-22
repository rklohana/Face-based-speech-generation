"""Extract a consistent 16 kHz, 80-band log-mel target from VoxCeleb2 audio.

Requires: matching torch and torchaudio builds, plus numpy.
The mel configuration is written next to the output manifest. Train the
vocoder with this exact feature definition, or use a compatible checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.prepare_manifests import read_jsonl, write_jsonl


MEL_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "f_min": 0.0,
    "f_max": 8000.0,
    "power": 1.0,
    "center": True,
    "pad_mode": "reflect",
    "norm": "slaney",
    "mel_scale": "slaney",
    "log_floor": 1e-5,
}


def extract(rows: list[dict], output_dir: Path) -> list[dict]:
    import torch
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(**{
        key: value for key, value in MEL_CONFIG.items() if key != "log_floor"
    })
    output = []
    for row in rows:
        audio_path = Path(row["audio_path"]).resolve()
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError(f"Empty or nonfinite waveform: {audio_path}")
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != MEL_CONFIG["sample_rate"]:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, MEL_CONFIG["sample_rate"]
            )
        if waveform.size(-1) < MEL_CONFIG["win_length"]:
            raise ValueError(f"Clip is shorter than the mel window: {audio_path}")
        with torch.no_grad():
            features = torch.log(mel(waveform).clamp_min(MEL_CONFIG["log_floor"]))
        if not torch.isfinite(features).all():
            raise ValueError(f"Nonfinite mel features: {audio_path}")
        digest = hashlib.sha256(str(audio_path).encode("utf-8")).hexdigest()[:20]
        destination = output_dir / row["speaker_id"] / f"{digest}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, features.squeeze(0).cpu().numpy().astype(np.float32))
        output.append({**row, "mel_path": str(destination.resolve()),
                       "mel_frames": features.size(-1)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Speech or reviewed transcript JSONL")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        parser.error("Input manifest is empty")
    output = extract(rows, args.output_dir)
    write_jsonl(args.output_manifest, output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mel_config.json").write_text(
        json.dumps(MEL_CONFIG, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(output)} mel arrays")


if __name__ == "__main__":
    main()
