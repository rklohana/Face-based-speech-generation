from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
import time

class Spectrograms(data.Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.train_dataset = []
        self.preprocess()
        self.num_images = len(self.train_dataset)
        self.shuffled = np.arange(self.num_images)
        np.random.shuffle(self.shuffled)

    def preprocess(self):
        num_embeds = 10
        embeddings = np.load('../embeddings.npy', allow_pickle=True)
        ids = np.load('../ids.npy', allow_pickle=True)
        speaker2idx = {}
        for idx in range(len(ids)):
            key = ids[idx][2:]
            if key in speaker2idx:
                speaker2idx[key].append(idx)
            else:
                speaker2idx[key] = [idx]
        data_dir = '../voxceleb/dev/aac'
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            start = time.time()
            voices = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.npy'):
                        voices.append(os.path.join(root, file))
            for voice in voices:
                voice_id = folder_name[2:]
                embed_id = voice_id
                if folder_name[2:] in speaker2idx:
                    ems = speaker2idx[folder_name[2:]]
                    for em in ems[0:num_embeds]:
                        embedding = embeddings[em]
                        self.train_dataset.append([voice_id, embed_id, voice, embedding])
            print("Time taken is {}".format(time.time() - start))

    def __getitem__(self, index):
        dataset = self.train_dataset
        if self.mode == "test":
            voiceid, _, file, _ = dataset[index]
            _, embedId, _, label = dataset[self.shuffled[index]]
        else:
            voiceid, embedId, file, label = dataset[index]
        spec = np.load(file)
        if spec.shape[1] < 300:
            spec = np.pad(spec, ((0, 0), (0, 300 - spec.shape[1])), mode='wrap')
        elif spec.shape[1] > 300:
            spec = spec[:, :300]
        assert(spec.shape[1] == 300)
        return voiceid, embedId, torch.FloatTensor(spec), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images

def get_loader(image_size=256, batch_size=8, mode='train', num_workers=4):
    if mode == "test":
        batch_size = 1
    spec_dataset = Spectrograms(mode)
    data_loader = data.DataLoader(dataset=spec_dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == "train"),
                                  num_workers=num_workers)
    return data_loader
