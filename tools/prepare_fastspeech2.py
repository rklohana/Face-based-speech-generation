"""Build FastSpeech2 targets from reviewed text, MFA TextGrids, and log-mels.

The output has phone tokens and measured phone durations plus frame-level
pitch and energy. MFA must have aligned the reviewed .lab text first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from tools.extract_mels import MEL_CONFIG
from tools.prepare_manifests import read_jsonl, write_jsonl


SILENCE = {"", "sil", "sp", "spn", "<eps>", "#"}


def read_phone_intervals(path: Path) -> list[tuple[float, float, str]]:
    """Read the `phones` tier from a long-format Praat TextGrid."""
    current_tier = None
    in_interval = False
    fields = {}
    intervals = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if re.fullmatch(r"item \[\d+\]:", line):
            current_tier = None
            in_interval = False
        elif line.startswith("name = "):
            current_tier = line.split('"', 2)[1] if '"' in line else None
        elif re.fullmatch(r"intervals \[\d+\]:", line):
            in_interval = current_tier == "phones" or str(current_tier).endswith(" - phones")
            fields = {}
        elif in_interval and " = " in line:
            key, value = line.split(" = ", 1)
            fields[key] = value
            if key == "text":
                if not {"xmin", "xmax", "text"}.issubset(fields):
                    raise ValueError(f"Incomplete phone interval in {path}")
                start, end = float(fields["xmin"]), float(fields["xmax"])
                if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                    raise ValueError(f"Invalid phone interval in {path}")
                label = value.strip('"').replace('""', '"')
                intervals.append((start, end, label))
                in_interval = False
    if not intervals:
        raise ValueError(f"No phones tier in TextGrid: {path}")
    return intervals


def durations_from_intervals(intervals: list[tuple[float, float, str]],
                             frames: int, audio_seconds: float) -> tuple[list[str], np.ndarray]:
    if frames < 1 or audio_seconds <= 0:
        raise ValueError("Invalid mel or audio duration")
    if intervals[0][0] > 0.1 or abs(intervals[-1][1] - audio_seconds) > 0.1:
        raise ValueError("Phone alignment does not cover the audio")
    for before, after in zip(intervals, intervals[1:]):
        if abs(before[1] - after[0]) > 0.03:
            raise ValueError("Phone alignment has a gap or overlap")
    tokens = []
    durations = []
    previous = 0
    total = intervals[-1][1]
    for index, (_, end, label) in enumerate(intervals):
        boundary = frames if index == len(intervals) - 1 else min(frames, max(previous, round(end / total * frames)))
        duration = boundary - previous
        previous = boundary
        phone = "sp" if label.strip().lower() in SILENCE else label.strip().upper()
        if duration == 0:
            if phone != "sp":
                raise ValueError(f"Phone {phone} quantizes to zero mel frames")
            continue
        tokens.append(phone)
        durations.append(duration)
    result = np.asarray(durations, dtype=np.int64)
    if not tokens or int(result.sum()) != frames:
        raise ValueError("Phone durations do not sum to mel frames")
    return tokens, result


def read_mono_wav(path: Path, target_sample_rate: int) -> np.ndarray:
    sample_rate, samples = wavfile.read(path)
    if samples.dtype == np.uint8:
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / float(np.iinfo(samples.dtype).max)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if not np.isfinite(samples).all() or len(samples) == 0:
        raise ValueError(f"Empty or nonfinite WAV: {path}")
    if sample_rate != target_sample_rate:
        factor = math.gcd(sample_rate, target_sample_rate)
        samples = signal.resample_poly(samples, target_sample_rate // factor,
                                       sample_rate // factor).astype(np.float32)
    return samples


def frame_pitch_energy(samples: np.ndarray, frames: int, sample_rate: int,
                       hop: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation F0 and RMS on mel-centered windows.

    Pitch is log1p(Hz)/7 for voiced frames and zero for unvoiced frames.
    Energy is log1p(100 * RMS). These definitions must match inference.
    """
    padded = np.pad(samples, (window // 2, window // 2), mode="reflect")
    pitch = np.zeros(frames, dtype=np.float32)
    energy = np.zeros(frames, dtype=np.float32)
    min_lag = max(1, math.floor(sample_rate / 400))
    max_lag = min(window - 1, math.ceil(sample_rate / 60))
    for index in range(frames):
        start = index * hop
        chunk = padded[start:start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        energy[index] = math.log1p(100 * rms)
        if rms < 0.005:
            continue
        centered = chunk - float(np.mean(chunk))
        corr = signal.correlate(centered, centered, mode="full", method="fft")[window - 1:]
        power = np.cumsum(centered * centered)
        left = power[window - 1 - np.arange(min_lag, max_lag + 1)]
        right = power[-1] - power[np.arange(min_lag, max_lag + 1) - 1]
        score = corr[min_lag:max_lag + 1] / np.sqrt(np.maximum(left * right, 1e-12))
        best = int(np.argmax(score))
        if score[best] >= 0.35:
            pitch[index] = math.log1p(sample_rate / (min_lag + best)) / 7.0
    return pitch, energy


def prepare(rows: list[dict], mels: list[dict], centroids: list[dict],
            output_dir: Path) -> tuple[list[dict], dict[str, int]]:
    mel_by_audio = {str(Path(row["audio_path"]).resolve()): row for row in mels}
    centroid_by_speaker = {row["speaker_id"]: row for row in centroids}
    if len(mel_by_audio) != len(mels) or len(centroid_by_speaker) != len(centroids):
        raise ValueError("Duplicate mel or centroid records")
    output = []
    vocabulary = {"<pad>", "sp"}
    for row in rows:
        if row.get("transcript_status") != "verified":
            raise ValueError("FastSpeech2 requires reviewed transcripts")
        audio_path = Path(row["audio_path"]).resolve()
        mel_row = mel_by_audio.get(str(audio_path))
        centroid = centroid_by_speaker.get(row["speaker_id"])
        if mel_row is None or centroid is None:
            raise ValueError(f"Missing mel or ECAPA centroid for {audio_path}")
        if mel_row["split"] != row["split"] or centroid["split"] != row["split"]:
            raise ValueError(f"Split mismatch for {audio_path}")
        mel = np.load(mel_row["mel_path"], allow_pickle=False)
        if mel.ndim != 2 or mel.shape[0] != MEL_CONFIG["n_mels"] or not np.isfinite(mel).all():
            raise ValueError(f"Invalid mel: {mel_row['mel_path']}")
        samples = read_mono_wav(audio_path, MEL_CONFIG["sample_rate"])
        intervals = read_phone_intervals(Path(row["alignment_path"]))
        phones, durations = durations_from_intervals(
            intervals, mel.shape[1], len(samples) / MEL_CONFIG["sample_rate"]
        )
        pitch, energy = frame_pitch_energy(
            samples, mel.shape[1], MEL_CONFIG["sample_rate"],
            MEL_CONFIG["hop_length"], MEL_CONFIG["win_length"]
        )
        digest = hashlib.sha256(str(audio_path).encode("utf-8")).hexdigest()[:20]
        target = output_dir / row["speaker_id"] / f"{digest}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, phones=np.asarray(phones), durations=durations,
                            pitch=pitch, energy=energy)
        vocabulary.update(phones)
        output.append({**row, "mel_path": mel_row["mel_path"],
                       "centroid_path": centroid["centroid_path"],
                       "features_path": str(target.resolve()),
                       "mel_frames": int(mel.shape[1])})
    if not output:
        raise ValueError("No aligned TTS records")
    return output, {phone: index for index, phone in enumerate(["<pad>"] + sorted(vocabulary - {"<pad>"}))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-index", required=True, type=Path)
    parser.add_argument("--mels", required=True, type=Path)
    parser.add_argument("--centroids", required=True, type=Path)
    parser.add_argument("--mel-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    args = parser.parse_args()
    if json.loads(args.mel_config.read_text(encoding="utf-8")) != MEL_CONFIG:
        raise ValueError("Mel configuration differs from the checked feature contract")
    rows, vocab = prepare(read_jsonl(args.alignment_index), read_jsonl(args.mels),
                          read_jsonl(args.centroids), args.output_dir)
    write_jsonl(args.output_manifest, rows)
    args.vocabulary.parent.mkdir(parents=True, exist_ok=True)
    args.vocabulary.write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows)} aligned FastSpeech2 records and {len(vocab)} phone symbols")


if __name__ == "__main__":
    main()
