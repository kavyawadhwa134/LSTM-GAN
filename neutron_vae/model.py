# model.py
import torch
import torch.nn as nn
from config import *

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(
            INPUT_DIM + COND_DIM, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.fc_mu = nn.Linear(HIDDEN_SIZE, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_SIZE, LATENT_DIM)

        self.decoder = nn.LSTM(
            LATENT_DIM + COND_DIM, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.fc_out = nn.Linear(HIDDEN_SIZE, INPUT_DIM)

    def encode(self, x, cond):
        c = cond.unsqueeze(1).repeat(1, x.size(1), 1)
        x_cond = torch.cat([x, c], dim=-1)
        _, (h, _) = self.encoder(x_cond)
        h = h[-1]  # last layer hidden state
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        c = cond.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        z_seq = z.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        z_cond = torch.cat([z_seq, c], dim=-1)
        out, _ = self.decoder(z_cond)
        return torch.tanh(self.fc_out(out))  # [-1,1]

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar, beta=0.1):
    mse = torch.mean((recon - x) ** 2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld