"""model.py"""
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_MNIST(nn.Module):
    def __init__(self, z_dim=10, l=3, nc=1):
        """builds VAE from https://github.com/casey-meehan/data-copying/
        Inputs: 
            - d: dimension of latent space 
            - l: number of layers 
        """
        super(BetaVAE_MNIST, self).__init__()
        self.z_dim = z_dim
        assert nc == 1
        
        #Build VAE here 
        self.encoder, self.decoder = self.build_VAE(z_dim, l)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
        
    def build_VAE(self, d, l): 
        """builds VAE with specified latent dimension and number of layers 
        Inputs: 
            -d: latent dimension 
            -l: number of layers 
        """
        encoder_layers = []
        decoder_layers = []
        alpha = 3 / l 

        for lyr in range(l)[::-1]:
            lyr += 1
            dim_a = int(np.ceil(2**(alpha*(lyr+1))))
            dim_b = int(np.ceil(2**(alpha*lyr)))
            if lyr == l: 
                encoder_layers.append(nn.Linear(784, d * dim_b))
                encoder_layers.append(nn.ReLU())
                decoder_layers.insert(0, nn.Linear(d * dim_b, 784))
                decoder_layers.insert(0, nn.ReLU())
            else: 
                encoder_layers.append(nn.Linear(d * dim_a, d * dim_b))
                encoder_layers.append(nn.ReLU())
                decoder_layers.insert(0, nn.Linear(d * dim_b, d * dim_a))
                decoder_layers.insert(0, nn.ReLU())
        encoder_layers.insert(0, View((-1, 784)))
        encoder_layers.append(nn.Linear(d*int(np.ceil(2**(alpha))), 2*d))
        decoder_layers.insert(0, nn.Linear(d, d*int(np.ceil(2**(alpha))) ))
        decoder_layers.append(View((-1, 1, 28, 28)))

        encoder = nn.Sequential(*encoder_layers)
        decoder = nn.Sequential(*decoder_layers)

        return encoder, decoder
        
    def _encode(self, x):
        """take an image, and return latent space mean + log variance
        Inputs: 
            -images, x, flattened to 784
        Outputs: 
            -means in latent dimension
            -logvariances in latent dimension 
        """
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        """Sample in latent space according to mean and logvariance
        Inputs: 
            -mu: batch of means
            -logvar: batch of logvariances
        Outputs: 
            -samples: batch of latent samples 
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decode(self, z):
        """Decode latent space samples
        Inputs: 
            -z: batch of latent samples 
        Outputs: 
            -x_recon: batch of reconstructed images 
        """
        raw_out = self.decoder(z)
        return raw_out
        # return torch.sigmoid(raw_out)

    def forward(self, x):
        """Do full encode and decode of images
        Inputs: 
            - x: batch of images 
        Outputs: 
            - recon_x: batch of reconstructed images
            - mu: batch of latent mean values 
            - logvar: batch of latent logvariances 
        """
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        return self._decode(z), mu, logvar


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass
