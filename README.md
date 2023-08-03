<div align="center">

  <img src="https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/c56d1547-7ece-413b-a13b-ccd70665b785" alt="logo" width="120px" height="120px" height="auto" />
  <br/>
  <h1>Facial features manipulation using VAE</h1>

  <h3>
    PSIML 9 Project
  </h3>


  </div>
<br />

The goal of this project was to manipulate facial features (like beard and glasses) using Variational Autoencoder. <br>
The dataset used for training is CelebA. <br>
Motivation: https://arxiv.org/abs/1611.05507

### Variational Autoencoder

Autoencoder is a neural network designed to learn an identity function to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation.
It is made from two parts: an encoder and a decoder. <br>
The encoder takes input and maps it to low-dimensional latent space. <br>
Decoder takes that vector and decodes it back to the original image <br>
In the variational autoencoder, the encoder part doesn't map an input to a vector but to distribution, so latent space is filled better. <br>

![vae](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/faa70966-fee5-4be8-a58a-639f7d27b65f)

### Manipulating features in latent space


The main idea was to train VAE and after that calculate the average encoding of images with and without some feature. When we subtract those values we will get a vector by which we should translate the encoding of an image so after decoding we would get an image with or without that feature.


![Screenshot 2023-08-03 134717](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/67d349b7-b7e2-4bc8-bf3a-5b17e25f9af6)
![Screenshot 2023-08-03 134749](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/3bebf8c4-ab1b-4721-9d7c-1dc4b392ff46)

### Architecture


![Screenshot 2023-08-03 135048](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/e3fa4c43-fffb-42c3-875d-509be393d723)


### Loss function


Loss is constructed from two parts Reconstruction loss and KL-divergence loss.<br>
Reconstruction loss is penalizing the model for differences in input and output images.<br>
KL-divergence loss should bring distributions returned by the encoder closer to standard normal distribution.<br>

![Screenshot 2023-08-03 140731](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/3a8f741b-e1ab-49de-8f89-9123d7601607)


### Results


| Average person with and without beard  | Average person with and without glasses |
| :---: | :---: |
| ![Screenshot 2023-08-03 140926](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/3d7ad00e-f8b1-49ea-9438-861ed820a25b)| ![Screenshot 2023-08-03 140942](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/a9a33767-b987-405e-aa84-0ac60ae4e6ef)|
| Images are made by decoding average of encodings | 

<br>

![Screenshot 2023-08-03 140902](https://github.com/mihailot01/face-features-vae-psiml9/assets/71095081/3415efe1-70f2-4043-8d9c-001df71c7c5e)



