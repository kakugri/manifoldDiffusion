import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from sklearn.manifold import Isomap
import random
import numpy as np
import matplotlib.pyplot as plt
from model import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
isomap_dim = 256  # Reduced dimension using Isomap
image_size = 64
batch_size = 256
num_epochs = 50
lr = 0.01
subset_size = 25000 # Smaller subset for Isomap computation

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CelebA(root='/home/riddhish/manifold_diffusion/data/celeba', split='train', transform=transform, download=True)

# Subset for training
indices = list(range(len(dataset)))
random.shuffle(indices)
indices = indices[:subset_size]
dataset = Subset(dataset, indices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Precompute Isomap latents
print("Extracting flattened features...")
flat_features = []

for data, _ in dataloader:
    # Flatten images to vectors
    images = data.view(data.size(0), -1).numpy()  # Shape: [batch_size, image_size*image_size*3]
    flat_features.append(images)

flat_features = np.concatenate(flat_features, axis=0)  # Shape: [num_samples, image_size*image_size*3]

# Perform Isomap on flattened features
print("Performing Isomap dimensionality reduction...")
isomap = Isomap(n_components=isomap_dim, n_neighbors=50)
isomap_latents = isomap.fit_transform(flat_features)  # Shape: [num_samples, isomap_dim]

# Create a dataset of Isomap latents and original images
original_images = []

for data, _ in dataloader:
    original_images.append(data)

original_images = torch.cat(original_images, dim=0)  # Shape: [subset_size, 3, image_size, image_size]

assert len(original_images) == len(isomap_latents), \
    f"Mismatch in size: original_images({len(original_images)}) vs isomap_latents({len(isomap_latents)})"

isomap_latents = torch.tensor(isomap_latents, dtype=torch.float32)
isomap_dataset = torch.utils.data.TensorDataset(isomap_latents, original_images)
isomap_dataloader = DataLoader(isomap_dataset, batch_size=batch_size, shuffle=True)

# Decoder initialization
decoder = Decoder(isomap_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=lr)

# Training function
def train_decoder(decoder, dataloader, criterion, optimizer):
    decoder.train()
    total_loss = 0

    for batch_idx, (latent, images) in enumerate(dataloader):
        latent = latent.to(device)
        images = images.to(device)

        outputs = decoder(latent)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Visualization
def visualize_reconstruction(decoder, dataloader, epoch):
    decoder.eval()
    num_images = 8

    with torch.no_grad():
        latent, images = next(iter(dataloader))
        latent = latent[:num_images].to(device)
        images = images[:num_images].to(device)

        reconstructed = decoder(latent)

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
        plt.savefig(f'/home/riddhish/manifold_diffusion/images/isomap_epoch_{epoch}.png')

# Training loop
print("Start Training Decoder...")
for epoch in range(num_epochs):
    avg_loss = train_decoder(decoder, isomap_dataloader, criterion, optimizer)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    if (epoch + 1) % 10 == 0:
        visualize_reconstruction(decoder, isomap_dataloader, epoch)
