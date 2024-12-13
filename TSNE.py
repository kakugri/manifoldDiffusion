import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import random
import numpy as np
import matplotlib.pyplot as plt
from model import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
tsne_dim = 256  # Reduced dimension using t-SNE
image_size = 64
batch_size = 256
num_epochs = 50
lr = 0.01
subset_size = 25000  # Smaller subset for t-SNE computation

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

# Precompute t-SNE latents
print("Extracting flattened features...")
flat_features = []

for data, _ in dataloader:
    # Flatten images to vectors
    images = data.view(data.size(0), -1).numpy()  # Shape: [batch_size, image_size*image_size*3]
    flat_features.append(images)

flat_features = np.concatenate(flat_features, axis=0)  # Shape: [num_samples, image_size*image_size*3]
# Perform t-SNE on flattened features
print("Performing t-SNE dimensionality reduction...")
tsne_dim = min(tsne_dim, 3)  # Ensure t-SNE dimensionality is valid
tsne = TSNE(n_components=tsne_dim, perplexity=30, n_iter=1000, random_state=42, method='barnes_hut')
tsne_latents = tsne.fit_transform(flat_features)  # Shape: [num_samples, tsne_dim]


# Create a dataset of t-SNE latents and original images
original_images = []

for data, _ in dataloader:
    original_images.append(data)

original_images = torch.cat(original_images, dim=0)  # Shape: [subset_size, 3, image_size, image_size]

assert len(original_images) == len(tsne_latents), \
    f"Mismatch in size: original_images({len(original_images)}) vs tsne_latents({len(tsne_latents)})"

tsne_latents = torch.tensor(tsne_latents, dtype=torch.float32)
tsne_dataset = torch.utils.data.TensorDataset(tsne_latents, original_images)
tsne_dataloader = DataLoader(tsne_dataset, batch_size=batch_size, shuffle=True)

# Decoder initialization
decoder = Decoder(tsne_dim).to(device)

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
        plt.savefig(f'/home/riddhish/manifold_diffusion/images/tsne_epoch_{epoch}.png')

# Training loop
print("Start Training Decoder...")
for epoch in range(num_epochs):
    avg_loss = train_decoder(decoder, tsne_dataloader, criterion, optimizer)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    if (epoch + 1) % 10 == 0:
        visualize_reconstruction(decoder, tsne_dataloader, epoch)
