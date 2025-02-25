#!/usr/bin/env python
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.tensorboard import SummaryWriter

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Global constant: target duration in seconds for each clip
TARGET_SECONDS = 8

def clip_or_pad(audio_dict, target_seconds=TARGET_SECONDS):
    """
    Clips or pads the input audio (a dict with keys "array" and "sampling_rate")
    to exactly `target_seconds` seconds.
    """
    audio = audio_dict["array"]
    sr = audio_dict["sampling_rate"]
    target_length = int(sr * target_seconds)
    if len(audio) > target_length:
        # For training, you might randomly choose a starting point:
        start = random.randint(0, len(audio) - target_length)
        audio = audio[start:start + target_length]
    elif len(audio) < target_length:
        pad_length = target_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode="constant")
    return audio

# A simple character-level tokenizer
VOCAB = {ch: idx + 1 for idx, ch in enumerate("abcdefghijklmnopqrstuvwxyz '")}
VOCAB_SIZE = len(VOCAB) + 1  # reserve index 0 for padding

def tokenize(text):
    """
    Converts a text string into a list of token ids.
    """
    text = text.lower()
    return [VOCAB.get(ch, 0) for ch in text]

def pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value] * (max_len - len(seq))

class VoxCelebTTSDataset(Dataset):
    """
    Dataset class for fine-tuning a TTS model on VoxCeleb2.
    Since the original dataset does not contain transcripts,
    we create a dummy transcript using the speaker id.
    Each sample is preprocessed:
      - Audio is clipped/padded to 8 seconds.
      - A mel-spectrogram is computed as the target.
    """
    def __init__(self, split="train", target_seconds=TARGET_SECONDS):
        self.dataset = load_dataset("acul3/voxceleb2", split=split)
        self.target_seconds = target_seconds
        # We assume 16kHz audio; adjust sample_rate if necessary.
        self.mel_transform = MelSpectrogram(sample_rate=16000, n_mels=80)
        self.db_transform = AmplitudeToDB()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_dict = sample["audio"]
        processed_audio = clip_or_pad(audio_dict, self.target_seconds)
        # Convert to torch tensor and add a channel dimension
        waveform = torch.tensor(processed_audio, dtype=torch.float).unsqueeze(0)  # (1, L)
        # Compute mel spectrogram (in decibels)
        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)
        # Create a dummy transcript using the speaker id (or "unknown" if missing)
        speaker_id = sample.get("speaker_id", "unknown")
        transcript = f"This is speaker {speaker_id} speaking."
        token_ids = tokenize(transcript)
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "mel_spec": mel_spec.squeeze(0),  # (n_mels, time)
            "speaker_id": speaker_id
        }

def collate_fn(batch):
    """
    Collate function for batching:
      - Pads token sequences to the length of the longest sequence.
      - Pads mel spectrograms along the time axis.
      - Converts speaker ids to integer values.
    """
    # Pad token sequences
    token_seqs = [item["token_ids"] for item in batch]
    max_token_len = max([len(seq) for seq in token_seqs])
    token_batch = torch.stack([
        torch.tensor(pad_sequence(seq.tolist(), max_token_len)) for seq in token_seqs
    ])
    
    # Pad mel spectrograms along time axis
    mel_specs = [item["mel_spec"] for item in batch]
    n_mels = mel_specs[0].shape[0]
    max_mel_time = max([mel.shape[1] for mel in mel_specs])
    mel_batch = []
    for mel in mel_specs:
        pad_len = max_mel_time - mel.shape[1]
        if pad_len > 0:
            pad = torch.zeros(n_mels, pad_len)
            mel_padded = torch.cat([mel, pad], dim=1)
        else:
            mel_padded = mel
        mel_batch.append(mel_padded)
    mel_batch = torch.stack(mel_batch)  # (batch, n_mels, time)
    
    # Convert speaker id to integer (if not already numeric)
    speaker_ids = []
    for item in batch:
        spk = item["speaker_id"]
        try:
            spk_int = int(spk)
        except:
            spk_int = 0  # default or use a mapping in a full implementation
        speaker_ids.append(spk_int)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
    
    return token_batch, mel_batch, speaker_ids

# ---------------------------------------------------------------------------
# Model Components: A simplified FastSpeech2-style TTS model
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=4, num_heads=4, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, seq_len)
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, embed_dim)
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0, 1)  # back to (batch, seq_len, embed_dim)
        return out

class DurationPredictor(nn.Module):
    def __init__(self, embed_dim, filter_size=256, kernel_size=3, dropout=0.1):
        super(DurationPredictor, self).__init__()
        self.conv1 = nn.Conv1d(embed_dim, filter_size, kernel_size, padding=kernel_size // 2)
        self.relu1 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(filter_size)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size // 2)
        self.relu2 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(filter_size)
        self.dropout2 = nn.Dropout(dropout)
        self.linear = nn.Linear(filter_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.transpose(1, 2)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.transpose(1, 2)
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        out = self.linear(x)  # (batch, seq_len, 1)
        out = out.squeeze(-1)  # (batch, seq_len)
        # Ensure positive durations; add 1.0 to avoid zeros.
        out = torch.relu(out) + 1.0
        return out

class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()
    
    def forward(self, encodings, durations):
        """
        Replicates encoder outputs according to predicted durations.
        Note: This is a simplified implementation.
        """
        output = []
        for i in range(encodings.size(0)):
            rep = []
            for j in range(encodings.size(1)):
                # Round duration prediction to an integer number of frames
                repeat = int(round(durations[i, j].item()))
                rep.append(encodings[i, j:j+1].expand(repeat, -1))
            if rep:
                output.append(torch.cat(rep, dim=0))
            else:
                # In case of an empty sequence
                output.append(torch.zeros((1, encodings.size(2)), device=encodings.device))
        # Pad sequences to the maximum length in the batch
        max_len = max(o.size(0) for o in output)
        out_padded = []
        for o in output:
            if o.size(0) < max_len:
                pad = torch.zeros((max_len - o.size(0), o.size(1)), device=o.device)
                o = torch.cat([o, pad], dim=0)
            out_padded.append(o)
        out_padded = torch.stack(out_padded)  # (batch, max_len, embed_dim)
        return out_padded

class MelDecoder(nn.Module):
    def __init__(self, embed_dim, n_mels, num_layers=4, num_heads=4, dropout=0.1):
        super(MelDecoder, self).__init__()
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_dim, n_mels)
    
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        x = self.decoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        mel_out = self.linear(x)  # (batch, seq_len, n_mels)
        mel_out = mel_out.transpose(1, 2)  # (batch, n_mels, time)
        return mel_out

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_mels, num_speakers, speaker_embed_dim,
                 num_encoder_layers=4, num_decoder_layers=4, num_heads=4, dropout=0.1):
        super(FastSpeech2, self).__init__()
        self.text_encoder = TextEncoder(vocab_size, embed_dim, num_layers=num_encoder_layers,
                                        num_heads=num_heads, dropout=dropout)
        self.duration_predictor = DurationPredictor(embed_dim, dropout=dropout)
        self.length_regulator = LengthRegulator()
        self.mel_decoder = MelDecoder(embed_dim, n_mels, num_layers=num_decoder_layers,
                                      num_heads=num_heads, dropout=dropout)
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embed_dim)
        # Project speaker embedding to the same dimension as the text encoder output
        self.spk_proj = nn.Linear(speaker_embed_dim, embed_dim)
    
    def forward(self, text_input, speaker_ids, src_key_padding_mask=None):
        # text_input: (batch, seq_len)
        encodings = self.text_encoder(text_input, src_key_padding_mask=src_key_padding_mask)
        # Obtain speaker embedding and project
        spk_emb = self.speaker_embedding(speaker_ids)  # (batch, speaker_embed_dim)
        spk_emb = self.spk_proj(spk_emb)  # (batch, embed_dim)
        # Add the speaker embedding to each time step of the text encoding
        encodings = encodings + spk_emb.unsqueeze(1)
        # Predict token durations
        durations = self.duration_predictor(encodings)  # (batch, seq_len)
        # Length regulation: repeat encoder outputs according to durations
        regulated = self.length_regulator(encodings, durations)
        # Decode mel spectrogram from regulated representations
        mel_output = self.mel_decoder(regulated)  # (batch, n_mels, time)
        return mel_output, durations

def compute_loss(mel_pred, mel_target, durations_pred):
    mel_loss = nn.MSELoss()(mel_pred, mel_target)
    batch_size, _, mel_time = mel_target.shape
   
    dummy_target = []
    for i in range(batch_size):
        text_len = (mel_target[i].sum(dim=0) != 0).sum().float()
        target = mel_time / (text_len + 1e-6)
        dummy_target.append(target)
    dummy_target = torch.tensor(dummy_target, device=mel_target.device).unsqueeze(1)
    durations_target = dummy_target.expand_as(durations_pred)
    duration_loss = nn.L1Loss()(durations_pred, durations_target)
    return mel_loss + duration_loss

def train(model, dataloader, optimizer, scheduler, device, num_epochs, log_dir="logs"):
    writer = SummaryWriter(log_dir=log_dir)
    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        for batch_idx, (token_batch, mel_batch, speaker_ids) in enumerate(dataloader):
            token_batch = token_batch.to(device)   
            mel_batch = mel_batch.to(device)         
            speaker_ids = speaker_ids.to(device)     
            
            optimizer.zero_grad()
            mel_pred, durations_pred = model(token_batch, speaker_ids)
            loss = compute_loss(mel_pred, mel_batch, durations_pred)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if global_step % 10 == 0:
                print(f"Epoch [{epoch+1}], Step [{global_step}], Loss: {loss.item():.4f}")
                writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1
        scheduler.step()
        
        checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs and checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--num_speakers", type=int, default=1000,
                        help="Total number of speakers (adjust based on dataset statistics)")
    args = parser.parse_args()
    
    
    dataset = VoxCelebTTSDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    
    embed_dim = 256
    n_mels = 80
    speaker_embed_dim = 128
    
    model = FastSpeech2(vocab_size=VOCAB_SIZE, embed_dim=embed_dim, n_mels=n_mels,
                        num_speakers=args.num_speakers, speaker_embed_dim=speaker_embed_dim)
    model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train(model, dataloader, optimizer, scheduler, args.device, args.num_epochs, log_dir=args.log_dir)

if __name__ == "__main__":
    main()
