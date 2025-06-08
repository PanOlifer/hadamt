import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


class MLPVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar


def train_vae(X, epochs=5, device="cpu"):
    dataset = TensorDataset(torch.tensor(X).float())
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = MLPVAE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for epoch in range(epochs):
        total = 0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = mse(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item()
        logger.info(f"vae epoch {epoch+1} loss {total/len(loader):.4f}")
    return model


def reconstruction_error(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X).float().to(device)
        recon, _, _ = model(X)
        return ((X - recon) ** 2).mean(dim=1).cpu().numpy()
