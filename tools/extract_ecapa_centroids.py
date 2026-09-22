"""Extract normalized ECAPA-TDNN speaker targets from a speech manifest.

Requires matching torch/torchaudio builds, speechbrain, and numpy. The default
SpeechBrain checkpoint emits 192-dimensional embeddings from 16 kHz audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.prepare_manifests import read_jsonl, write_jsonl


DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000
EMBEDDING_DIM = 192


def unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != EMBEDDING_DIM or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite {EMBEDDING_DIM}-dimensional ECAPA vector")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        raise ValueError("ECAPA returned a zero speaker vector")
    return values / norm


def extract(rows: list[dict], encoder, output_dir: Path,
            min_utterances: int = 1) -> list[dict]:
    import torch
    import torchaudio

    if min_utterances < 1:
        raise ValueError("min_utterances must be positive")
    speakers: dict[str, dict] = {}
    for row in rows:
        speaker = row["speaker_id"]
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {speaker}: {split}")
        if speaker in speakers and speakers[speaker]["split"] != split:
            raise ValueError(f"Speaker {speaker} occurs in multiple splits")
        speakers.setdefault(speaker, {"split": split, "audio_paths": []})["audio_paths"].append(row["audio_path"])

    output = []
    for speaker, record in sorted(speakers.items()):
        vectors = []
        for audio_path in record["audio_paths"]:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.numel() == 0 or not torch.isfinite(waveform).all():
                raise ValueError(f"Empty or nonfinite audio: {audio_path}")
            waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
            with torch.no_grad():
                embedding = encoder.encode_batch(waveform).detach().cpu().numpy()
            vectors.append(unit_vector(embedding))
        if len(vectors) < min_utterances:
            continue
        centroid = unit_vector(np.mean(vectors, axis=0))
        destination = output_dir / f"{speaker}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, centroid)
        output.append({"speaker_id": speaker, "split": record["split"],
                       "centroid_path": str(destination.resolve()),
                       "utterances": len(vectors)})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-utterances", type=int, default=3)
    args = parser.parse_args()
    rows = read_jsonl(args.speech)
    if not rows:
        parser.error("Speech manifest is empty")
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(source=args.source, run_opts={"device": args.device})
    output = extract(rows, encoder, args.output_dir, args.min_utterances)
    if not output:
        raise ValueError("No speaker has enough utterances for a centroid")
    write_jsonl(args.output_manifest, output)
    (args.output_dir / "ecapa_config.json").write_text(json.dumps({
        "source": args.source,
        "sample_rate": SAMPLE_RATE, "embedding_dim": EMBEDDING_DIM,
        "normalization": "unit utterances, mean, unit centroid",
        "min_utterances": args.min_utterances,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output)} ECAPA speaker centroids")


if __name__ == "__main__":
    main()
