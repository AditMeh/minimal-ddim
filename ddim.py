import argparse

from diffusers import UNet2DModel
from torchvision.datasets import MNIST

import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

import tqdm

def compute_schedule(T, beta_min, beta_max, device):
    betas = torch.linspace(beta_min, beta_max, steps=T, device=device)

    alphas = 1 - betas
    sqrt_alphas = torch.sqrt(alphas)
    var_t = torch.sqrt(betas)

    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    schedule_hparams = {
        "beta_t": betas,
        "var_t": var_t,
        "alphas": alphas,
        "sqrt_alphas": sqrt_alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "oneover_sqrta": 1/torch.sqrt(alphas),
        "mab_over_sqrtmab": (1-alphas)/sqrt_one_minus_alpha_bar
    }
    return schedule_hparams


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def sample_backward_ddim(schedule, T, seed, unet, device):
    with torch.no_grad():
        for i in tqdm.tqdm(range(T, 0, -1)):
            ts = torch.ones(1).to(device) * i

            pred_eps = unet(seed, ts.float())['sample']

            pred_x0 = (1 / schedule["sqrt_alpha_bar"][i-1]) * (seed -
                                                             (schedule["sqrt_one_minus_alpha_bar"][i-1]) * pred_eps)
            term1 = schedule["sqrt_alpha_bar"][i-2] * pred_x0 if i > 1 else pred_x0
            
            term2 = schedule["sqrt_one_minus_alpha_bar"][i-2] * pred_eps if i > 1 else 0
            # term3 = z * schedule["var_t"][i-1]

            seed = term1 + term2
            
        return rescale(seed)

def sample_forward_ddim(batch, T, schedule, unet, device):
    with torch.no_grad():
        seed = batch
        for i in tqdm.tqdm(range(0, T, 1)):
            ts = torch.ones(1).to(device) * i
            
            if i == 0:
                seed = seed + ((1- torch.sqrt(1/schedule["alpha_bar"][0])) * seed + (torch.sqrt(1/schedule["alpha_bar"][0] -1))*unet(seed, ts.float())['sample'])*torch.sqrt(schedule["alpha_bar"][0])
            else:        
                pred_eps = unet(seed, ts.float())['sample']
                
                term1 = (torch.sqrt(1/schedule["alpha_bar"][i-1])- torch.sqrt(1/schedule["alpha_bar"][i])) * seed
                
                term2 = (torch.sqrt(1/schedule["alpha_bar"][i] - 1) - torch.sqrt(1/schedule["alpha_bar"][i-1] - 1)) * pred_eps

                seed = seed + torch.sqrt(schedule["alpha_bar"][i]) * (term1 + term2)

        return seed

if __name__ == "__main__":
    T = 1000
    beta_min = 1e-4
    beta_max = 0.02
    epochs = 30
    lr = 0.0001
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', nargs='?', type=str, help='checkpoint')
    args = parser.parse_args()

    device = (torch.device('cuda')
              if torch.cuda.is_available() else torch.device('cpu'))

    unet = UNet2DModel(sample_size=(32, 32), in_channels=1, out_channels=1).to(device)

    unet.load_state_dict(torch.load(args.checkpoint))


    train_dataset = MNIST(
        "./data",
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32, shuffle=True, num_workers=4)

    def extract_sample(x): return x[0] if len(x) > 1 else x
    batch = extract_sample(next(iter(dataloader))).to(device)
    save_image(batch, "original.png")

    unet = unet.to(device)

    schedule = compute_schedule(T, beta_min, beta_max, device)

    seed = torch.randn(*batch.shape).to(device=device)

    i = sample_forward_ddim(batch, T, schedule, unet, device)
    save_image(i, "ddim_forward.png")

    i = sample_backward_ddim(schedule, T, i, unet, device)

    save_image(i, "ddim_backward.png")
