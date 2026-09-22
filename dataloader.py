from torch.utils import data
import torch
import os
import numpy as np

class Spectrograms(data.Dataset):
    def __init__(self, mode, data_dir, embeddings_path, ids_path, image_size=256):
        if mode not in ("train", "test"):
            raise ValueError("mode must be 'train' or 'test'")
        if image_size < 1:
            raise ValueError("image_size must be positive")
        self.mode = mode
        self.data_dir = data_dir
        self.image_size = image_size
        self.embeddings_path = embeddings_path
        self.ids_path = ids_path
        self.train_dataset = []
        self.preprocess()
        self.num_images = len(self.train_dataset)
        self.shuffled = np.arange(self.num_images)
        np.random.shuffle(self.shuffled)

    def preprocess(self):
        num_embeds = 10
        embeddings = np.load(self.embeddings_path)
        ids = np.load(self.ids_path)
        if embeddings.ndim != 2 or ids.ndim != 1:
            raise ValueError("Expected 2D embeddings and 1D IDs arrays")
        if len(embeddings) != len(ids):
            raise ValueError("embeddings and ids must have the same number of rows")
        speaker2idx = {}
        for idx in range(len(ids)):
            key = str(ids[idx])[2:]
            if key in speaker2idx:
                speaker2idx[key].append(idx)
            else:
                speaker2idx[key] = [idx]
        for folder_name in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            voices = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.npy'):
                        voices.append(os.path.join(root, file))
            for voice in sorted(voices):
                voice_id = folder_name[2:]
                embed_id = voice_id
                if voice_id in speaker2idx:
                    ems = speaker2idx[voice_id]
                    for em in ems[0:num_embeds]:
                        embedding = embeddings[em]
                        self.train_dataset.append([voice_id, embed_id, voice, embedding])
        if not self.train_dataset:
            raise ValueError("No matching spectrograms and embeddings were found")

    def __getitem__(self, index):
        dataset = self.train_dataset
        if self.mode == "test":
            voiceid, _, file, _ = dataset[index]
            _, embedId, _, label = dataset[self.shuffled[index]]
        else:
            voiceid, embedId, file, label = dataset[index]
        spec = np.load(file)
        if spec.ndim != 2 or min(spec.shape) == 0:
            raise ValueError(f"Expected a nonempty 2D spectrogram: {file}")
        height, width = spec.shape
        spec = np.pad(spec[:self.image_size, :self.image_size],
                      ((0, max(0, self.image_size - height)),
                       (0, max(0, self.image_size - width))), mode='constant')
        return voiceid, embedId, torch.as_tensor(spec, dtype=torch.float32).unsqueeze(0), torch.as_tensor(label, dtype=torch.float32)

    def __len__(self):
        return self.num_images

def get_loader(data_dir, embeddings_path, ids_path, image_size=256, batch_size=8, mode='train', num_workers=4):
    if mode == "test":
        batch_size = 1
    spec_dataset = Spectrograms(mode, data_dir, embeddings_path, ids_path, image_size)
    data_loader = data.DataLoader(dataset=spec_dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == "train"),
                                  num_workers=num_workers)
    return data_loader
