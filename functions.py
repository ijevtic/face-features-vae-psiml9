from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import CustomDataset

def get_avg_mu(model, labels):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 32
    glasses_dataset = CustomDataset("data/img_align_celeba/",labels,transforms.ToTensor())
    glasses_loader = DataLoader(glasses_dataset, batch_size=batch_size, shuffle=True)

    mu_sum = torch.Tensor(1000)
    sigma_sum = torch.Tensor(1000)
    count = 0
    for batch in glasses_loader:
        count += batch.shape[0]
        batch = batch.to(device)
        mu, sigma = model.encode(batch)
        # print(mu[0][:5])
        mu_sum = mu_sum + mu.T.sum(dim=1)
        sigma_sum = sigma_sum + sigma.T.sum(dim=1)
        # print(sigma_sum.shape)
    
    # print(count)
    mu_avg = mu_sum/count
    sigma_avg = sigma_sum/count

    # print(sigma_sum.shape)
    return mu_avg