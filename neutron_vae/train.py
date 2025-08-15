# train.py
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import UltraHighFidelityVAE, vae_loss
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
from neutron_vae.config import *

# Load and preprocess
raw_tracks = load_and_split_tracks(CSV_FILE)
tracks_norm, conditions, xyz_min, xyz_max = preprocess_tracks(raw_tracks)

# Convert to torch
tracks_norm = torch.tensor(tracks_norm, dtype=torch.float32).to(DEVICE)
conditions = torch.tensor(conditions, dtype=torch.float32).to(DEVICE)
tracks_norm = 2 * tracks_norm - 1  # [0,1] → [-1,1] for tanh

# Model
model = UltraHighFidelityVAE().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

# Advanced learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2000, T_mult=2, eta_min=LR/1000
)

# Checkpoint loading functionality
def load_checkpoint(checkpoint_path):
    """Load checkpoint and continue training from where it left off"""
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get starting epoch and best loss
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        
        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Starting from epoch: {start_epoch}")
        print(f"   Best loss so far: {best_loss:.6f}")
        
        return start_epoch, best_loss
    else:
        print(f"📝 No checkpoint found at {checkpoint_path}, starting fresh training")
        return 0, float('inf')

# Load checkpoint if available
start_epoch, best_loss = load_checkpoint('neutron_vae_checkpoint.pth')

# Beta scheduling for KL divergence
def get_beta(epoch):
    if epoch < WARMUP_EPOCHS:
        return 0.0  # No KL loss during warmup
    else:
        # Gradually increase beta
        progress = (epoch - WARMUP_EPOCHS) / (N_EPOCHS - WARMUP_EPOCHS)
        return BETA * min(1.0, progress * 3)  # Slower increase

# Training
print("🚀 Starting Ultra High-Fidelity VAE training...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training for {N_EPOCHS} epochs with {BATCH_SIZE} batch size")
print(f"Learning rate: {LR}, Beta: {BETA}, Alpha: {ALPHA}")

loss_history = []
beta_history = []

for epoch in range(start_epoch, N_EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Dynamic beta scheduling
    current_beta = get_beta(epoch) if BETA_SCHEDULE else BETA
    beta_history.append(current_beta)

    # Random batch
    idx = torch.randperm(len(tracks_norm))[:BATCH_SIZE]
    x_batch = tracks_norm[idx]
    c_batch = conditions[idx]

    recon, mu, logvar = model(x_batch, c_batch)
    loss, loss_components = vae_loss(recon, x_batch, mu, logvar, beta=current_beta, alpha=ALPHA)

    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
    
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss_components['total'])

    # Detailed logging
    if epoch % 500 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:4d}: Loss = {loss_components['total']:.6f} "
              f"(MSE: {loss_components['mse']:.6f}, "
              f"KL: {loss_components['kld']:.6f}, "
              f"β: {current_beta:.4f}, "
              f"LR: {lr:.6f})")
    
    # Save checkpoint every 1000 epochs
    if epoch % 1000 == 0 and epoch > 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss_components['total'],
            'best_loss': best_loss,
            'config': {
                'latent_dim': LATENT_DIM,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS
            }
        }, 'neutron_vae_checkpoint.pth')
        print(f"💾 Checkpoint saved at epoch {epoch}")
    
    # Save best model
    if loss_components['total'] < best_loss:
        best_loss = loss_components['total']
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': best_loss,
            'config': {
                'latent_dim': LATENT_DIM,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS
            }
        }, 'neutron_vae_best.pth')

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': N_EPOCHS,
    'loss': loss_history[-1],
    'config': {
        'latent_dim': LATENT_DIM,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS
    }
}, 'neutron_vae.pth')

print(f"✅ Training complete!")
print(f"Best loss: {best_loss:.6f}")
print(f"Final loss: {loss_history[-1]:.6f}")
print(f"Final beta: {beta_history[-1]:.6f}")
print("Models saved: neutron_vae.pth (final), neutron_vae_best.pth (best), neutron_vae_checkpoint.pth (checkpoint)")

# Plot training progress
try:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss history
    ax1.plot(loss_history)
    ax1.set_title('Ultra High-Fidelity Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Beta history
    ax2.plot(beta_history)
    ax2.set_title('Beta Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Beta')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('ultra_training_progress.png', dpi=150, bbox_inches='tight')
    print("✅ Ultra training progress saved as 'ultra_training_progress.png'")
except ImportError:
    print("Matplotlib not available for plotting training progress")