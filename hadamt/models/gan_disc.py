import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, latent_dim=16, output_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_gan(X, epochs=3, device="cpu"):
    dataset = TensorDataset(torch.tensor(X).float())
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    G = Generator(output_dim=X.shape[1]).to(device)
    D = Discriminator(input_dim=X.shape[1]).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
    bce = nn.BCELoss()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            size = batch.size(0)
            real = torch.ones(size, 1, device=device)
            fake = torch.zeros(size, 1, device=device)
            # D
            opt_D.zero_grad()
            loss_real = bce(D(batch), real)
            z = torch.randn(size, 16, device=device)
            fake_data = G(z)
            loss_fake = bce(D(fake_data.detach()), fake)
            (loss_real + loss_fake).backward()
            opt_D.step()
            # G
            opt_G.zero_grad()
            loss_g = bce(D(fake_data), real)
            loss_g.backward()
            opt_G.step()
    return G, D


def disc_score(D, X, device="cpu"):
    D.eval()
    with torch.no_grad():
        X = torch.tensor(X).float().to(device)
        prob = D(X).squeeze().cpu().numpy()
    return 1.0 - prob
