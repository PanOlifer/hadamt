import torch
from torch import nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Simple convolutional VAE for MNIST sized images."""

    def __init__(self, latent_dim=16, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(x, recon_x, mu, logvar, mse=True):
    if mse:
        recon = F.mse_loss(recon_x, x, reduction="sum")
    else:
        recon = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


def train_vae(model, dataloader, optimizer, device, epochs=5, mse=True):
    model.train()
    for _ in range(epochs):
        total = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(x, recon, mu, logvar, mse)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"loss {total / len(dataloader.dataset):.4f}")


def reconstruction_errors(model, dataloader, device, mse=True):
    model.eval()
    errs = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            recon, _, _ = model(x)
            if mse:
                rec = F.mse_loss(recon, x, reduction="none")
            else:
                rec = F.binary_cross_entropy(recon, x, reduction="none")
            rec = rec.view(rec.size(0), -1).mean(dim=1)
            errs.extend(rec.cpu().tolist())
    return errs
