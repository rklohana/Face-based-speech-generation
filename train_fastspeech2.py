"""Train speaker-conditioned FastSpeech2 on reviewed, MFA-aligned speech.

Input is produced by tools.prepare_fastspeech2. The old TTS_finetuning.py
baseline is not used. This command needs torch, numpy, and actual training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fastspeech2_model import DEFAULT_CONFIG, FaceConditionedFastSpeech2, fastspeech2_loss
from tools.extract_ecapa_centroids import EMBEDDING_DIM, unit_vector
from tools.extract_mels import MEL_CONFIG
from tools.prepare_manifests import read_jsonl


class AlignedSpeechDataset(Dataset):
    def __init__(self, rows: list[dict], vocabulary: dict[str, int]):
        self.rows = rows
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get("transcript_status") != "verified":
            raise ValueError("FastSpeech2 cannot train on an unreviewed transcript")
        mel = np.load(row["mel_path"], allow_pickle=False).astype(np.float32)
        speaker = unit_vector(np.load(row["centroid_path"], allow_pickle=False))
        with np.load(row["features_path"], allow_pickle=False) as features:
            phones = features["phones"].tolist()
            durations = features["durations"].astype(np.int64)
            pitch = features["pitch"].astype(np.float32)
            energy = features["energy"].astype(np.float32)
        try:
            phone_ids = np.asarray([self.vocabulary[phone] for phone in phones], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Phone absent from vocabulary: {error}") from error
        if (mel.ndim != 2 or mel.shape[0] != MEL_CONFIG["n_mels"] or
                len(phones) != len(durations) or int(durations.sum()) != mel.shape[1] or
                len(pitch) != mel.shape[1] or len(energy) != mel.shape[1] or
                not np.isfinite(mel).all() or not np.isfinite(pitch).all() or
                not np.isfinite(energy).all()):
            raise ValueError(f"Invalid FastSpeech2 target: {row['audio_path']}")
        return phone_ids, durations, mel, pitch, energy, speaker


def collate(batch):
    batch_size = len(batch)
    max_phones = max(len(item[0]) for item in batch)
    max_frames = max(item[2].shape[1] for item in batch)
    phones = torch.zeros(batch_size, max_phones, dtype=torch.long)
    durations = torch.zeros_like(phones)
    mels = torch.zeros(batch_size, MEL_CONFIG["n_mels"], max_frames)
    pitch = torch.zeros(batch_size, max_frames)
    energy = torch.zeros_like(pitch)
    speakers = torch.zeros(batch_size, EMBEDDING_DIM)
    for index, (phone_ids, counts, mel, f0, rms, speaker) in enumerate(batch):
        phones[index, :len(phone_ids)] = torch.from_numpy(phone_ids)
        durations[index, :len(counts)] = torch.from_numpy(counts)
        mels[index, :, :mel.shape[1]] = torch.from_numpy(mel)
        pitch[index, :len(f0)] = torch.from_numpy(f0)
        energy[index, :len(rms)] = torch.from_numpy(rms)
        speakers[index] = torch.from_numpy(speaker)
    return phones, durations, mels, pitch, energy, speakers


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    for batch in loader:
        phones, durations, mels, pitch, energy, speaker = [item.to(device) for item in batch]
        with torch.set_grad_enabled(training):
            output = model(phones, speaker, durations=durations, pitch=pitch, energy=energy)
            losses = fastspeech2_loss(output, mels, durations, pitch, energy, phones)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += float(losses["total"].detach()) * len(phones)
        samples += len(phones)
    return total / samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--mel-config", required=True, type=Path)
    parser.add_argument("--ecapa-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Training hyperparameters must be positive")
    mel_config = json.loads(args.mel_config.read_text(encoding="utf-8"))
    if mel_config != MEL_CONFIG:
        raise ValueError("Mel configuration differs from the checked feature contract")
    ecapa_config = json.loads(args.ecapa_config.read_text(encoding="utf-8"))
    if ecapa_config.get("embedding_dim") != EMBEDDING_DIM:
        raise ValueError("ECAPA configuration has an incompatible embedding dimension")
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    if vocabulary.get("<pad>") != 0 or set(vocabulary.values()) != set(range(len(vocabulary))):
        raise ValueError("Phone vocabulary needs contiguous IDs with pad=0")
    rows = read_jsonl(args.manifest)
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    if not training or not validation:
        raise ValueError("Need train and validation TTS utterances")
    if {row["speaker_id"] for row in training} & {row["speaker_id"] for row in validation}:
        raise ValueError("TTS train and validation speakers overlap")
    train_loader = DataLoader(AlignedSpeechDataset(training, vocabulary),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(AlignedSpeechDataset(validation, vocabulary),
                            batch_size=args.batch_size, collate_fn=collate)
    model = FaceConditionedFastSpeech2(len(vocabulary), DEFAULT_CONFIG).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, args.device)
        val_loss = run_epoch(model, val_loader, None, args.device)
        print(f"epoch {epoch + 1}: train {train_loss:.4f}, val {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "model_config": model.config,
                        "vocabulary": vocabulary, "mel_config": mel_config,
                        "ecapa_config": ecapa_config,
                        "best_val_loss": best_loss, "epoch": epoch + 1}, args.output)
    print(f"Saved best model at {args.output}")


if __name__ == "__main__":
    main()
