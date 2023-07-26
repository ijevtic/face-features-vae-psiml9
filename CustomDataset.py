from torch.utils.data import Dataset
import os
import torch
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_dir, img_labels, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image =  Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return torch.flatten(image)