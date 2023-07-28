
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img_path = "data/img_align_celeba"
df = pd.read_csv("data/list_attr_celeba.csv")
# df = df.head(10000)
df = df.tail(10000)
print(df.columns)
with_glasses = df['image_id'][(df['Eyeglasses'] == 1) & (df['Male']==1)].tolist()
without_glasses = df['image_id'][(df['Eyeglasses'] == -1) & (df['Male']==1)].tolist()

print(len(with_glasses))
print(len(without_glasses))

print(with_glasses[-5])


# glass_images = [Image.open(f'{img_path}/{with_glasses[np.random.randint(0, len(with_glasses))]}') for i in range(5)]
# f = plt.figure()
# for i in range(5):
#     # Debug, plot figure
#     f.add_subplot(1, 5, i + 1)
#     plt.axis('off')
#     plt.imshow(glass_images[i])

# plt.show(block=True) 

no_glass_images = [Image.open(f'{img_path}/{without_glasses[np.random.randint(0, len(without_glasses))]}') for i in range(5)]
f = plt.figure()
for i in range(5):
    # Debug, plot figure
    f.add_subplot(1, 5, i + 1)
    plt.axis('off')
    plt.imshow(no_glass_images[i])

plt.show(block=True) 

from VanillaVAE import VanillaVAE
import torch
from torchvision import transforms


INPUT_DIM = 3
Z_DIM = 1000
PATH = "model_prvenac.pt"
device = torch.device('cpu')

model = VanillaVAE(INPUT_DIM, Z_DIM)
checkpoint = torch.load(PATH,map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

transform=transforms.ToTensor()

# f = plt.figure()
# for i in range(len(glass_images)):
#     with torch.no_grad():
#         out = model.forward(transform(glass_images[i]).unsqueeze(0).to(device))[0]
        
#     out = out.view(-1, 3, 224, 192)
#     out = np.transpose(out, (0, 2, 3, 1))
#     f.add_subplot(1, len(glass_images), i + 1)
#     plt.axis('off')
#     plt.imshow(out[0])


from torchvision import transforms

from functions import get_avg_mu
 
mu_avg_glasses = get_avg_mu(model, with_glasses)
# mu_avg_no_glasses = get_avg_mu(without_glasses)
mu_avg_no_glasses = get_avg_mu(model, without_glasses[:len(with_glasses)])

delta_mu = mu_avg_glasses - mu_avg_no_glasses
print(delta_mu.shape)

#delta_mu = torch.load('delta_mu.pt')

import matplotlib.pyplot as plt


f = plt.figure()
for i in range(len(no_glass_images)):
    with torch.no_grad():
        x = transform(no_glass_images[i]).unsqueeze(0).to(device)
        # x = torch.rand_like(transform(no_glass_images[i]).unsqueeze(0)).to(device) 
        # x = torch.zeros_like(x).to(device)
        out = model.generate_with_delta(x,delta_mu*4) #torch.zeros_like(delta_mu))
        
    out = out.view(-1, 3, 224, 192)
    out = np.transpose(out, (0, 2, 3, 1))
    f.add_subplot(1, len(no_glass_images), i + 1)
    plt.axis('off')
    plt.imshow(out[0])
