import random
import imageio
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA

class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=1000, min_beta=10 ** -4, max_beta = 0.02, device = None, image_chw=(3,64,64)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i+1]) for i in range(len(self.alphas))]).to(device)
        
    def forward(self, x0, t, eta=None):
        
        n,c,h,w = x0.shape
        a_bar = self.alpha_bars[t]
        
        if eta is None:
            eta = torch.randn(n,c,h,w).to(self.device)
            
        noisy = a_bar.sqrt().reshape(n,1,1,1) * x0 + (1 - a_bar).sqrt().reshape(n,1,1,1) * eta
        return noisy
    
    def backward(self, x, t):
        return self.network(x,t)
    
    
def sinusoidal_embedding(n,d):
    embedding = torch.zeros(n,d)
    wk = torch.tensor([1/1000 ** (2 * j /d) for j in range(d)])
    wk = wk.reshape((1,d))
    t = torch.arange(n).reshape((n,1))
    embedding[:,::2] = torch.sin(t*wk[:,::2])
    embedding[:,1::2] = torch.cos(t*wk[:,::2])
    
    return embedding

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm([in_c, *shape[1:]])  # Adjust for in_c dynamically
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize
        
    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 64, 64), 1, 30),
            MyBlock((64, 64, 64), 30, 30),
            MyBlock((30, 64, 64), 30, 30)
            # MyBlock((n_channels, image_size, image_size), n_channels, 64),
            # MyBlock((64, image_size, image_size), 64, 64),
            # MyBlock((64, image_size, image_size), 64, 64)
        )
        self.down1 = nn.Conv2d(30, 30, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 30)
        self.b2 = nn.Sequential(
            MyBlock((30, 32, 32), 30, 60),
            MyBlock((60, 32, 32), 60, 60),
            MyBlock((60, 32, 32), 60, 60)
        )
        self.down2 = nn.Conv2d(60, 60, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 60)
        self.b3 = nn.Sequential(
            MyBlock((60, 16, 16), 60, 120),
            MyBlock((120, 16, 16), 120, 120),
            MyBlock((120, 16, 16), 120, 120)
        )
        self.down3 = nn.Conv2d(120, 120, 4, 2, 1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 120)

        self.b_mid = nn.Sequential(
            MyBlock((120, 8, 8), 120, 60),
            MyBlock((60, 8, 8), 60, 60),
            MyBlock((60, 8, 8), 60, 120)
        )

        # Second half
        self.up1 = nn.ConvTranspose2d(120, 120, 4, 2, 1)

        self.te4 = self._make_te(time_emb_dim, 240)
        self.b4 = nn.Sequential(
            MyBlock((240, 16, 16), 240, 120),
            MyBlock((120, 16, 16), 120, 60),
            MyBlock((60, 16, 16), 60, 60)
        )

        self.up2 = nn.ConvTranspose2d(60, 60, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 120)
        self.b5 = nn.Sequential(
            MyBlock((120, 32, 32), 120, 60),
            MyBlock((60, 32, 32), 60, 30),
            MyBlock((30, 32, 32), 30, 30)
        )

        self.up3 = nn.ConvTranspose2d(30, 30, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 60)
        self.b_out = nn.Sequential(
            MyBlock((60, 64, 64), 60, 30),
            MyBlock((30, 64, 64), 30, 30),
            MyBlock((30, 64, 64), 30, 30, normalize=False)
        )

        self.conv_out = nn.Conv2d(30, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 64, 64) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 64, 64)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 32, 32)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 16, 16)
        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 16, 16)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 16, 16)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 32, 32)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 32, 32)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 64, 64)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 64, 64)

        out = self.conv_out(out)

        return out

    
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

         
def generate_new_images(ddpm, n_samples=1, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=64, w=64):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Showing the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x

# Training
store_path = "ddpm_celeba.pt" 
def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0])
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps = 350
# Loading the trained model
best_model = MyDDPM(MyUNet(), n_steps=350, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")

print("Generating new images")
generated = generate_new_images(
        best_model,
        n_samples=6,
        device=device,
        gif_name="celeba.gif" )
from  torchvision.utils import save_image
save_image(generated, 'celeba.png')
print("done")

show_images(generated, "Final result")