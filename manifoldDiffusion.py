import torch 
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random
from KingdomModel import Encoder, Decoder, ManifoldEncoder
import imageio
from argparse import ArgumentParser

from tqdm.auto import tqdm

import einops

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Hyperparameters

latent_dim = 784
image_size = 28
batch_size = 128
num_epochs = 21
lr = 0.001
subset_size = 12800
test_size = subset_size + 100

fc = nn.Linear(latent_dim, 224 * 2 * 2)


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CelebA(root='/home/riddhish/manifold_diffusion/data/celeba', split='train', transform=transform, download=True)

indices = list(range(len(dataset)))
random.shuffle(indices)
indices = indices[:subset_size]
dataset = Subset(dataset, indices)
print(len(dataset))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
trainloader_manifold = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=2)

encoder = ManifoldEncoder()

# toLearn = next(iter(trainloader_manifold))[0].to(device)
for x, y in trainloader_manifold:
    toLearn = x
manifold_data = encoder(toLearn)
# manifold_data = fc(manifold_data)
manifold_data = manifold_data
labels = torch.tensor([0 for i in range(len(dataset))])
trainset_manifold = TensorDataset(manifold_data, labels)
trainloader_manifold = DataLoader(trainset_manifold, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

decoder = Decoder(latent_dim).to(device)

criterion = nn.MSELoss()

# optimizer = optim.Adam(list(encoder.parameters())+ list(decoder.parameters()), lr = lr)
optimizer = optim.Adam(decoder.parameters(), lr = lr)

def roundup(x):
    return math.ceil(x / 10.0) * 10

def train(encoder, decoder, dataloader, trainloader_manifold, criterion, optimizer):
    decoder.train()
    total_loss = 0

    for data, manifoldData in zip(dataloader, trainloader_manifold):
        
        real_images, _ = data  # We only need the images
        manifold_images, _ = manifoldData  # We only need the images
        
        real_images = real_images.to(device)
        # manifold_images = fc(manifold_images)
        # manifold_images = manifold_images.view(manifold_images.size(0), 1, 28, 28)
        manifold_images = manifold_images.to(device)

        #Forward Pass
        outputs = decoder(manifold_images)
        # print(outputs.shape)

        loss = criterion(outputs, real_images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/ len(dataloader)

def visualize_reconstruction(encoder, decoder, dataloader, trainloader_manifold, epoch):
    # encoder.eval()
    decoder.eval()
    num_images=8
    
    with torch.no_grad():
        images = next(iter(dataloader))[0][:num_images].to(device)
        manifold_images= next(iter(trainloader_manifold))[0][:num_images]
        # manifold_images = fc(manifold_images)
        # manifold_images = manifold_images.view(manifold_images.size(0), 224, 2, 2)
        manifold_images = manifold_images.to(device)
        reconstructed = decoder(manifold_images)
        
        # Convert images to CPU and denormalize
        images = images.cpu() * 0.5 + 0.5
        reconstructed = reconstructed.cpu() * 0.5 + 0.5
        
        plt.figure(figsize=(15, 4))
        for i in range(num_images):
            # Original images
            plt.subplot(2, num_images, i + 1)
            plt.imshow(images[i].permute(1, 2, 0))
            plt.axis('off')
            
            # Reconstructed images
            plt.subplot(2, num_images, i + num_images + 1)
            plt.imshow(reconstructed[i].permute(1, 2, 0))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'/home/riddhish/manifold_diffusion/images/kingdom_spectral_{epoch}.png')
        
print("Start Training..")
for epoch in range(num_epochs):
    avg_loss = train(encoder, decoder, dataloader, trainloader_manifold, criterion, optimizer)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # if (epoch + 1) % 5 == 0:
    #     visualize_reconstruction(encoder, decoder, dataloader, epoch)
    if epoch == num_epochs - 1:
        visualize_reconstruction(encoder, decoder, dataloader, trainloader_manifold, epoch)
        


#Diffusion part

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"

no_train = False
fashion = True
batch_size = 128
n_epochs = 20
lr = 0.01

store_path = "kingdom_dif.pt" if fashion else "ddpm_mnist.pt"

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
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()
    plt.savefig(f'/home/riddhish/manifold_diffusion/images/{title}.png')

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0].view(128, 1, 28, 28), "Images in the first batch")
        break

# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)
# ds_fn = FashionMNIST if fashion else MNIST
# dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
# loader = DataLoader(dataset, batch_size, shuffle=True)

dataset = TensorDataset(manifold_data, labels)
loader = DataLoader(trainset_manifold, batch_size=batch_size, shuffle=False, num_workers=2)

# Optionally, show a batch of regular images
# show_first_batch(loader)

# # Getting device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs = batch[0].view(128, 1, 28, 28)

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=1000, gif_name="sampling.gif", c=1, h=28, w=28):
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
                # print(normalized.shape, n_samples)
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

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
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
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

# Defining model
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

print(sum([p.numel() for p in ddpm.parameters()]))

# Optionally, show the diffusion (forward) process
show_forward(ddpm, loader, device)

# Optionally, show the denoising (backward) process
generated = generate_new_images(ddpm, gif_name="before_training.gif")
show_images(generated, "Images generated before training")

def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="kingdom_dif.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    #for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        #for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
        for step, batch in enumerate(loader):
            # Loading data
            x0 = batch[0].view(128, 1, 28, 28).to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

# Training
store_path = "kingdom_dif.pt" if fashion else "ddpm_mnist.pt"
if not no_train:
    training_loop(ddpm, loader, n_epochs, optim=optim.Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

# Loading the trained model
best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")

print("Generating new images")
generated = generate_new_images(
        best_model,
        n_samples=100,
        device=device,
        gif_name="kingdom_dif.gif" if fashion else "mnist.gif"
    )
print(generated.shape)
# generated = generated.view(100, 784)
# flat_features = torch.flatten(generated, 2, 3)
# flat_features = torch.flatten(flat_features, 1, 2)
# show_images(generated, "Final result")

# from IPython.display import Image

# Image(open('kingdom_dif.gif' if fashion else 'mnist.gif','rb').read())

with torch.no_grad():
    num_images=8
    images = next(iter(dataloader))[0][:num_images].to(device)
    manifold_images = generated[:num_images]
    # manifold_image = flat_features[:num_images]
    # manifold_images= next(iter(trainloader_manifold))[0][:num_images]
    # manifold_images = manifold_images.view(manifold_images.size(0), 224, 2, 2)
    manifold_images = manifold_images.to(device)
    reconstructed = decoder(manifold_images)
    
    # Convert images to CPU and denormalize
    images = images.cpu() * 0.5 + 0.5
    reconstructed = reconstructed.cpu() * 0.5 + 0.5
    
    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis('off')
        
        # Reconstructed images
        plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(reconstructed[i].permute(1, 2, 0))
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'/home/riddhish/manifold_diffusion/images/kingdom_ml_diffusion_{epoch}.png')

