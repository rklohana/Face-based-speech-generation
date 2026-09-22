"""Train a ResNet18 to predict frozen VoxCeleb2 ECAPA speaker centroids.

Requires matching torch/torchvision builds, Pillow, and numpy. Train and
validation speakers must be disjoint. The exported vectors share the 192-D
unit-normalized space of extract_ecapa_centroids.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.extract_ecapa_centroids import EMBEDDING_DIM, unit_vector
from tools.prepare_manifests import read_jsonl, write_jsonl


def make_model(pretrained: bool):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    width = backbone.fc.in_features
    backbone.fc = torch.nn.Linear(width, EMBEDDING_DIM)

    class NormalizedFaceEncoder(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, images):
            return torch.nn.functional.normalize(self.network(images), dim=-1)

    return NormalizedFaceEncoder(backbone)


def image_transform():
    from torchvision.models import ResNet18_Weights

    return ResNet18_Weights.DEFAULT.transforms()


def load_joined(faces_path: Path, centroids_path: Path):
    targets = {}
    for row in read_jsonl(centroids_path):
        speaker = row["speaker_id"]
        if speaker in targets:
            raise ValueError(f"Repeated ECAPA centroid for {speaker}")
        targets[speaker] = (row["split"], unit_vector(np.load(row["centroid_path"])))
    joined = []
    for row in read_jsonl(faces_path):
        speaker = row["speaker_id"]
        if speaker not in targets:
            raise ValueError(f"Face speaker has no ECAPA centroid: {speaker}")
        split, vector = targets[speaker]
        if split != row["split"]:
            raise ValueError(f"Face and speech splits differ for {speaker}")
        joined.append((row["image_path"], speaker, split, vector))
    if not joined:
        raise ValueError("No joined face and ECAPA targets")
    return joined


def train(args):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    joined = load_joined(args.faces, args.centroids)
    ecapa_config = json.loads(args.ecapa_config.read_text(encoding="utf-8"))
    if ecapa_config.get("embedding_dim") != EMBEDDING_DIM:
        raise ValueError("ECAPA configuration has an incompatible embedding dimension")
    training = [row for row in joined if row[2] == "train"]
    validation = [row for row in joined if row[2] == "val"]
    if not training or not validation:
        raise ValueError("Need face images from both train and val speakers")
    if len({row[1] for row in training}) < 2:
        raise ValueError("Need at least two training speakers for cosine-softmax learning")
    if {row[1] for row in training} & {row[1] for row in validation}:
        raise ValueError("Train and validation speakers overlap")
    transform = image_transform()

    class FaceRows(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            path, speaker, _, vector = self.records[index]
            with Image.open(path) as image:
                pixels = transform(image.convert("RGB"))
            return pixels, speaker, torch.from_numpy(vector.copy())

    train_ids = sorted({row[1] for row in training})
    train_targets = {speaker: next(row[3] for row in training if row[1] == speaker)
                     for speaker in train_ids}
    target_matrix = torch.tensor(np.stack([train_targets[s] for s in train_ids]),
                                 dtype=torch.float32, device=args.device)
    label_map = {speaker: index for index, speaker in enumerate(train_ids)}
    train_loader = DataLoader(FaceRows(training), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(FaceRows(validation), batch_size=args.batch_size,
                            num_workers=0)
    model = make_model(pretrained=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("-inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        count = 0
        for pixels, speakers, _ in train_loader:
            predicted = model(pixels.to(args.device))
            labels = torch.tensor([label_map[s] for s in speakers], device=args.device)
            logits = predicted @ target_matrix.T / args.temperature
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(speakers)
            count += len(speakers)
        model.eval()
        similarities = []
        with torch.no_grad():
            for pixels, _, target in val_loader:
                predicted = model(pixels.to(args.device))
                similarities.extend((predicted * target.to(args.device)).sum(dim=-1).cpu().tolist())
        val_cosine = float(np.mean(similarities))
        print(f"epoch {epoch + 1}: train loss {train_loss / count:.4f}, val positive cosine {val_cosine:.4f}")
        if val_cosine > best_val:
            best_val = val_cosine
            torch.save({"state_dict": model.state_dict(), "embedding_dim": EMBEDDING_DIM,
                        "backbone": "resnet18", "preprocess": "ResNet18_Weights.DEFAULT.transforms",
                        "ecapa_config": ecapa_config, "best_val_cosine": best_val}, args.output)


def embed(args):
    import torch
    from PIL import Image

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["embedding_dim"] != EMBEDDING_DIM or checkpoint["backbone"] != "resnet18":
        raise ValueError("Incompatible face checkpoint")
    model = make_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device).eval()
    transform = image_transform()
    output = []
    for row in read_jsonl(args.faces):
        with Image.open(row["image_path"]) as image:
            pixels = transform(image.convert("RGB")).unsqueeze(0).to(args.device)
        with torch.no_grad():
            vector = model(pixels).squeeze(0).cpu().numpy().astype(np.float32)
        vector = unit_vector(vector)
        destination = args.output_dir / row["speaker_id"] / (Path(row["image_path"]).stem + ".npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, vector)
        output.append({**row, "face_embedding_path": str(destination.resolve())})
    write_jsonl(args.output_manifest, output)
    print(f"Exported {len(output)} face embeddings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--faces", type=Path, required=True)
    fit.add_argument("--centroids", type=Path, required=True)
    fit.add_argument("--ecapa-config", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=10)
    fit.add_argument("--batch-size", type=int, default=32)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--temperature", type=float, default=0.07)
    fit.add_argument("--device", default="cpu")
    fit.set_defaults(func=train)
    export = commands.add_parser("embed")
    export.add_argument("--faces", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--output-manifest", type=Path, required=True)
    export.add_argument("--device", default="cpu")
    export.set_defaults(func=embed)
    args = parser.parse_args()
    if args.command == "train" and (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.temperature <= 0):
        parser.error("Training hyperparameters must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
