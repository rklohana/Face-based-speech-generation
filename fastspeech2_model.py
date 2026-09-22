"""Speaker-conditioned FastSpeech2-style acoustic model.

Uses phone durations from forced alignment and frame-level pitch/energy.
Speaker conditioning is a frozen, unit-normalized 192-D ECAPA vector, or a
face vector trained into the same space. This model predicts log-mels only.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CONFIG = {
    "hidden": 256,
    "heads": 4,
    "encoder_layers": 4,
    "decoder_layers": 4,
    "dropout": 0.1,
    "speaker_dim": 192,
    "n_mels": 80,
    "max_frames": 4096,
}


def positional_encoding(length: int, width: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    exponent = torch.arange(0, width, 2, device=device, dtype=torch.float32)
    scales = torch.exp(exponent * (-math.log(10000.0) / width))
    values = positions * scales
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = torch.sin(values)
    result[:, 1::2] = torch.cos(values[:, :result[:, 1::2].shape[1]])
    return result


class VariancePredictor(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.dropout(self.norm1(F.relu(self.conv1(values.transpose(1, 2)).transpose(1, 2))))
        values = self.dropout(self.norm2(F.relu(self.conv2(values.transpose(1, 2)).transpose(1, 2))))
        return self.output(values).squeeze(-1)


def regulate(encoded: torch.Tensor, durations: torch.Tensor,
             max_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    if durations.shape != encoded.shape[:2] or (durations < 0).any():
        raise ValueError("Durations must be nonnegative and match phone tokens")
    expanded = []
    lengths = []
    for sequence, counts in zip(encoded, durations):
        length = int(counts.sum().item())
        if length < 1 or length > max_frames:
            raise ValueError(f"Regulated sequence has {length} frames")
        expanded.append(torch.repeat_interleave(sequence, counts, dim=0))
        lengths.append(length)
    return nn.utils.rnn.pad_sequence(expanded, batch_first=True), torch.tensor(
        lengths, device=encoded.device, dtype=torch.long
    )


class FaceConditionedFastSpeech2(nn.Module):
    def __init__(self, vocab_size: int, config: dict | None = None):
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        hidden = self.config["hidden"]
        if hidden % self.config["heads"]:
            raise ValueError("hidden width must be divisible by attention heads")
        self.phone_embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.speaker_projection = nn.Linear(self.config["speaker_dim"], hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, self.config["heads"], hidden * 4, self.config["dropout"], batch_first=True
        )
        decoder_layer = nn.TransformerEncoderLayer(
            hidden, self.config["heads"], hidden * 4, self.config["dropout"], batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.config["encoder_layers"])
        self.decoder = nn.TransformerEncoder(decoder_layer, self.config["decoder_layers"])
        self.duration_predictor = VariancePredictor(hidden, self.config["dropout"])
        self.pitch_predictor = VariancePredictor(hidden, self.config["dropout"])
        self.energy_predictor = VariancePredictor(hidden, self.config["dropout"])
        self.pitch_embedding = nn.Linear(1, hidden)
        self.energy_embedding = nn.Linear(1, hidden)
        self.mel_projection = nn.Linear(hidden, self.config["n_mels"])
        self.postnet = nn.Sequential(
            nn.Conv1d(self.config["n_mels"], 256, 5, padding=2), nn.Tanh(),
            nn.Conv1d(256, self.config["n_mels"], 5, padding=2),
        )

    def forward(self, phones: torch.Tensor, speaker: torch.Tensor,
                durations: torch.Tensor | None = None,
                pitch: torch.Tensor | None = None,
                energy: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if phones.ndim != 2 or speaker.shape != (phones.size(0), self.config["speaker_dim"]):
            raise ValueError("Invalid phone or speaker embedding shape")
        if not torch.isfinite(speaker).all():
            raise ValueError("Nonfinite speaker embedding")
        if (torch.linalg.vector_norm(speaker, dim=-1) <= 1e-8).any():
            raise ValueError("Speaker embedding must be nonzero")
        phone_mask = phones.eq(0)
        if phone_mask.all(dim=1).any():
            raise ValueError("Every sample needs at least one phone")
        speaker = F.normalize(speaker, dim=-1)
        text = self.phone_embedding(phones) + positional_encoding(
            phones.size(1), self.config["hidden"], phones.device
        ).unsqueeze(0)
        text = text + self.speaker_projection(speaker).unsqueeze(1)
        encoded = self.encoder(text, src_key_padding_mask=phone_mask)
        log_duration = self.duration_predictor(encoded).masked_fill(phone_mask, 0.0)
        if durations is None:
            durations = torch.round(torch.expm1(log_duration.clamp(-2, 4))).long()
            durations = durations.clamp(1, 50).masked_fill(phone_mask, 0)
        elif (durations.masked_select(phone_mask) != 0).any():
            raise ValueError("Padded phone durations must be zero")
        expanded, mel_lengths = regulate(encoded, durations, self.config["max_frames"])
        mel_mask = torch.arange(expanded.size(1), device=phones.device).unsqueeze(0) >= mel_lengths.unsqueeze(1)
        pitch_pred = self.pitch_predictor(expanded).masked_fill(mel_mask, 0.0)
        energy_pred = self.energy_predictor(expanded).masked_fill(mel_mask, 0.0)
        if pitch is not None and pitch.shape != pitch_pred.shape:
            raise ValueError("Pitch target length does not match aligned mel frames")
        if energy is not None and energy.shape != energy_pred.shape:
            raise ValueError("Energy target length does not match aligned mel frames")
        varied = expanded + self.pitch_embedding((pitch if pitch is not None else pitch_pred).unsqueeze(-1))
        varied = varied + self.energy_embedding((energy if energy is not None else energy_pred).unsqueeze(-1))
        varied = varied + positional_encoding(expanded.size(1), self.config["hidden"], phones.device).unsqueeze(0)
        decoded = self.decoder(varied, src_key_padding_mask=mel_mask)
        mel = self.mel_projection(decoded).transpose(1, 2)
        refined = mel + self.postnet(mel)
        return {"mel": mel, "refined_mel": refined, "log_duration": log_duration,
                "pitch": pitch_pred, "energy": energy_pred,
                "durations": durations, "mel_lengths": mel_lengths}


def fastspeech2_loss(output: dict[str, torch.Tensor], target_mel: torch.Tensor,
                     target_durations: torch.Tensor, target_pitch: torch.Tensor,
                     target_energy: torch.Tensor, phones: torch.Tensor) -> dict[str, torch.Tensor]:
    lengths = output["mel_lengths"]
    if target_mel.shape != output["mel"].shape:
        raise ValueError("Target mel shape differs from model output")
    frame_mask = torch.arange(target_mel.size(-1), device=target_mel.device)[None, :] < lengths[:, None]
    phone_mask = phones.ne(0)
    mel_denom = frame_mask.sum().clamp_min(1) * target_mel.size(1)
    mel_loss = ((output["mel"] - target_mel).abs() * frame_mask[:, None, :]).sum() / mel_denom
    postnet_loss = ((output["refined_mel"] - target_mel).abs() * frame_mask[:, None, :]).sum() / mel_denom
    duration_loss = ((output["log_duration"] - torch.log1p(target_durations.float())) ** 2 * phone_mask).sum() / phone_mask.sum().clamp_min(1)
    pitch_loss = (((output["pitch"] - target_pitch) ** 2) * frame_mask).sum() / frame_mask.sum().clamp_min(1)
    energy_loss = (((output["energy"] - target_energy) ** 2) * frame_mask).sum() / frame_mask.sum().clamp_min(1)
    total = mel_loss + postnet_loss + duration_loss + pitch_loss + energy_loss
    return {"total": total, "mel": mel_loss, "postnet": postnet_loss,
            "duration": duration_loss, "pitch": pitch_loss, "energy": energy_loss}
