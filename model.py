import torch
import torch.nn as nn
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride = 2, padding = 1)

        self.fc = nn.Linear(256*4*4, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)


    def forward(self, x):
        x = self.leaky_relu(self.batchnorm32(self.conv1(x)))
        x = self.leaky_relu(self.batchnorm64(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm128(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm256(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)



    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.relu(self.batchnorm128(self.deconv1(x)))
        x = self.relu(self.batchnorm64(self.deconv2(x)))
        x = self.relu(self.batchnorm32(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x


class ISOMAP(nn.Module):
    def __init__(self):
        super(ISOMAP, self).__init__()
    
    def forward(self, toLearn):
        print("Extracting flattened features...")
        flat_features = torch.flatten(toLearn, 2, 3)
        flat_features = torch.flatten(flat_features, 1, 2)
        
        # Perform Isomap dimensionality reduction
        print("Performing Isomap dimensionality reduction...")
        isomap = Isomap(n_components=32, n_neighbors=5)
        manifold_data = isomap.fit_transform(flat_features.cpu().numpy())  # Convert to numpy for sklearn
        manifold_data = torch.tensor(manifold_data, dtype=torch.float32, device=toLearn.device)

        return manifold_data
    
from sklearn.manifold import Isomap

class ManifoldEncoder(nn.Module):
    def __init__(self):
        super(ManifoldEncoder, self).__init__()
    
    def forward(self, toLearn):
        print("Extracting flattened features...")
        flat_features = torch.flatten(toLearn, 2, 3)
        flat_features = torch.flatten(flat_features, 1, 2)
        
        # Apply Isomap on each flattened image
        print("Performing Isomap dimensionality reduction...")
        isomap = Isomap(n_components=784, n_neighbors=5)
        
        # Convert the tensor to a numpy array for Isomap
        flat_features_np = flat_features.cpu().detach().numpy()
        manifold_data = isomap.fit_transform(flat_features_np)  # Shape: (batch_size, n_components)
        
        # Convert back to a tensor
        manifold_data = torch.tensor(manifold_data, dtype=torch.float32).to(toLearn.device)

        return manifold_data
