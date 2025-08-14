# train.py
import torch
from model import VAE, vae_loss
from load_data import load_and_split_tracks, preprocess_tracks
from config import *

# Load and preprocess
raw_tracks = load_and_split_tracks(CSV_FILE)
tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(raw_tracks)

# Convert to torch
tracks_norm = torch.tensor(tracks_norm, dtype=torch.float32).to(DEVICE)
conditions = torch.tensor(conditions, dtype=torch.float32).to(DEVICE)
tracks_norm = 2 * tracks_norm - 1  # [0,1] → [-1,1] for tanh

# Model
model = VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
print("Starting VAE training...")
for epoch in range(N_EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Random batch
    idx = torch.randperm(len(tracks_norm))[:BATCH_SIZE]
    x_batch = tracks_norm[idx]
    c_batch = conditions[idx]

    recon, mu, logvar = model(x_batch, c_batch)
    loss = vae_loss(recon, x_batch, mu, logvar, beta=BETA)

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Save
torch.save(model.state_dict(), 'neutron_vae.pth')
print("✅ Training complete. Model saved.")