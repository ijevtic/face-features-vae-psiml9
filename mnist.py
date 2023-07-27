import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.datasets as datasets 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from CustomDataset import CustomDataset
from torch import Tensor
from torch.utils.data import Dataset
import os
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

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('test')
INPUT_DIM = 3
#38804
Z_DIM = 20
H_DIM = 2000
NUM_EPOCHS = 50
BATCH_SIZE = 32
LR_RATE = 3e-4

PATH = "model.pt"

print(device)

print('aa')

batch_size = 32
data_length = 202599
dataset = CustomDataset("img_align_celeba", [(str(i).rjust(6, '0')+".jpg") for i in range(1,data_length+1)], transform=transforms.ToTensor())
print('yy')
dataset_train, dataset_val = random_split(dataset, [int(data_length*0.8), data_length- int(data_length*0.8)])
print('xx')

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
print('bb')