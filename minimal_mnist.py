from torchvision.datasets import MNIST

from diffusers import UNet2DModel

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import torchvision

import torch
import json
import tqdm

T = 1000
beta_min = 1e-4
beta_max= 0.02
epochs = 400
lr = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def compute_schedule(T, beta_min, beta_max, device):
    betas = torch.linspace(beta_min, beta_max, steps=T, device=device)
    alphas = 1 - betas

    std_t = torch.sqrt(betas)
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

    schedule_hparams = {
        "std_t": std_t,
        "alphas": alphas,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "oneover_sqrta": 1/torch.sqrt(alphas),
        "mab_over_sqrtmab": (1-alphas)/sqrt_one_minus_alpha_bar
    }
    return schedule_hparams

def sample(T, schedule, img_shape, unet, device):
    with torch.no_grad():
        seed = torch.randn(*img_shape).to(device=device)
        for i in tqdm.tqdm(range(T, 0, -1)):
            z = torch.randn(*img_shape).to(device=device) if i > 1 else 0
            ts = torch.ones(1).to(device) * i

            pred_eps = unet(seed, ts.float())['sample']
            term1 = schedule["oneover_sqrta"][i-1]
            term2 = seed - (schedule["mab_over_sqrtmab"][i-1] * pred_eps)
            term3 = z * schedule["std_t"][i-1]

            seed = term1 * term2 + term3

        return rescale(seed)

def sample_ddim(T, schedule, img_shape, unet, device):
    with torch.no_grad():
        seed = torch.randn(*img_shape).to(device=device)
        for i in tqdm.tqdm(range(T, 0, -1)):
            ts = torch.ones(1).to(device) * i
            pred_eps = unet(seed, ts.float())['sample']

            pred_x0 = (1 / schedule["sqrt_alpha_bar"][i-1]) * (seed -
                                                             (schedule["sqrt_one_minus_alpha_bar"][i-1]) * pred_eps)
            
            term1 = schedule["sqrt_alpha_bar"][i-2] * pred_x0 if i > 1 else pred_x0
            term2 = schedule["sqrt_one_minus_alpha_bar"][i-2] * pred_eps if i > 1 else 0
            seed = term1 + term2
        return rescale(seed)
∏



train_dataset = MNIST('../mnist_data',
                            download=True,
                            train=True,
                            transform=transforms.Compose([
                                transforms.Resize(
                                    (32, 32)),
                                transforms.ToTensor()]))

train = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32, shuffle=True, num_workers=96)


unet = UNet2DModel(sample_size=(32, 32), in_channels=1, out_channels=1).to(device)
schedule = compute_schedule(T, beta_min, beta_max, device)
optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="minimal mnist",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
    name="mnist")

for epoch in tqdm.tqdm(range(1, epochs+1)):
    pbar = tqdm.tqdm(iter(train))
    acc, denom = 0, 0
    
    for x, _ in pbar:
        x = x.to(device)
        # Forward diffuse 
        ts = torch.randint(1, T + 1, (x.shape[0],), device=device)
        eps = torch.randn(*x.shape, device=device)


        
        x_pass = schedule["sqrt_alpha_bar"][ts - 1][..., None, None, None] * x + \
            schedule["sqrt_one_minus_alpha_bar"][ts - 1][..., None, None, None] * eps
            
        pred = unet(x_pass, ts.float())['sample']
        loss = torch.nn.MSELoss()(pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc += loss.item()
        denom += x.shape[0]
        pbar.set_description(str(loss.item()))
        
    wandb.log({"loss": acc/denom})

    torch.save(unet.state_dict(), f'mnist_minimal_{epoch}.pt')

    sampled_imgs = sample(T, schedule, [4] + list(x.shape[1:]), unet, device)
    sampled_imgs_ddim = sample_ddim(T, schedule, [4] + list(x.shape[1:]), unet, device)

    images = wandb.Image(
        torchvision.utils.make_grid(sampled_imgs), 
        caption="generated numbers ddpm"
        )
            
    wandb.log({"generated numbers ddpm": images})
    
    images = wandb.Image(
        torchvision.utils.make_grid(sampled_imgs_ddim), 
        caption="generated numbers ddim"
        )
            
    wandb.log({"generated numbers ddim": images})