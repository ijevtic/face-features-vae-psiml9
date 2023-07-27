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

        self.imgs = []
        for i in range(len(img_labels)):
            img_label = img_labels[i]
            if i%1000 == 0:
                print(i)
            #print(img_label)
            img_path = f'{self.img_dir}/{img_label}'
            image =  Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            self.imgs.append(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.imgs[idx]