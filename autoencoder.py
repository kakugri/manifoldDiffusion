import torch 
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
from model import Encoder, Decoder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Hyperparameters

latent_dim = 512
image_size = 64
batch_size = 256
num_epochs = 50
lr = 0.01
subset_size = 100000

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CelebA(root='/home/riddhish/manifold_diffusion/data/celeba', split='train', transform=transform, download=True)

indices = list(range(len(dataset)))
random.shuffle(indices)
indices = indices[:subset_size]
dataset = Subset(dataset, indices)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(list(encoder.parameters())+ list(decoder.parameters()), lr = lr)

def save_model(encoder, decoder, unet, ddpm, path='model_checkpoint_2.pth'):
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'unet_state_dict': unet.state_dict(),
        'ddpm_state_dict': ddpm.state_dict(),
        'latent_dim': latent_dim,
        'image_size': image_size
    }, path)
    print(f"Model saved to {path}")
    
    
def train(encoder, decoder, dataloader, criterion, optimizer):
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        
        images = data.to(device)

        #Forward Pass
        latent = encoder(images)
        outputs = decoder(latent)

        loss = criterion(outputs, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/ len(dataloader)

def visualize_reconstruction(encoder, decoder, dataloader, epoch):
    encoder.eval()
    decoder.eval()
    num_images=8
    
    with torch.no_grad():
        images = next(iter(dataloader))[0][:num_images].to(device)
        latent = encoder(images)
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
        plt.savefig(f'/home/riddhish/manifold_diffusion/images/{epoch}')
        
print("Start Training..")
for epoch in range(num_epochs):
    avg_loss = train(encoder, decoder, dataloader, criterion, optimizer)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    if (epoch + 1) % 5 == 0:
        visualize_reconstruction(encoder, decoder, dataloader, epoch)

