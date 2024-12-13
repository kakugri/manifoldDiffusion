import torch
import torch.nn as nn
from sklearn.manifold import SpectralEmbedding


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


# class Decoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Decoder, self).__init__()
        
#         self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
#         self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#         self.batchnorm32 = nn.BatchNorm2d(32)
#         self.batchnorm64 = nn.BatchNorm2d(64)
#         self.batchnorm128 = nn.BatchNorm2d(128)



#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(x.size(0), 256, 4, 4)
#         x = self.relu(self.batchnorm128(self.deconv1(x)))
#         x = self.relu(self.batchnorm64(self.deconv2(x)))
#         x = self.relu(self.batchnorm32(self.deconv3(x)))
#         x = self.tanh(self.deconv4(x))
#         return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(latent_dim, 196 * 2 * 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.batchnorm32 = nn.BatchNorm2d(32)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)



    def forward(self, x):
        x = torch.flatten(x, 2, 3)
        x = torch.flatten(x, 1, 2)
        x = self.fc(x)
        # x = x.view(x.size(0), 1, 28, 28)
        x = x.view(x.size(0), 196, 2, 2)
        x = self.relu(self.batchnorm128(self.deconv1(x)))
        x = self.relu(self.batchnorm64(self.deconv2(x)))
        x = self.relu(self.batchnorm32(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x

class ManifoldEncoder(nn.Module):
    def __init__(self):
        super(ManifoldEncoder, self).__init__()
        
        # self.fc = nn.Linear(784, 224 * 2 * 2)
    
    def forward(self, toLearn):
        print("Extracting flattened features...")
        flat_features = torch.flatten(toLearn, 2, 3)
        flat_features = torch.flatten(flat_features, 1, 2)
        
        # Apply Laplacian Eigenmaps on each flattened image
        print("Performing Laplacian Eigenmap dimensionality reduction...")
        embedding = SpectralEmbedding(n_components=784, affinity='nearest_neighbors', n_neighbors=150)
        manifold_data = embedding.fit_transform(flat_features)  # Shape: (batch_size, n_components)
        manifold_data = torch.tensor(manifold_data, dtype=torch.float32)
        # fc = nn.Linear(784, 224 * 2 * 2)
        # manifold_data = self.fc(manifold_data)
        manifold_data = manifold_data.view(manifold_data.size(0), 1, 28, 28)

        return manifold_data
    
