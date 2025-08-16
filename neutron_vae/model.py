# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.config import *

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, x):
        return x + self.layers(x)

class UltraHighFidelityVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder: Deep Bidirectional LSTM + Transformer
        self.encoder_lstm = nn.LSTM(
            INPUT_DIM + COND_DIM, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.1, bidirectional=True
        )
        
        # Multiple transformer layers for better sequence modeling
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(HIDDEN_SIZE * 2, num_heads=16, dropout=0.1)
            for _ in range(4)
        ])
        
        # Deep residual blocks for better feature extraction
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(HIDDEN_SIZE * 2, dropout=0.1)
            for _ in range(3)
        ])
        
        # Latent space projection with better initialization
        self.fc_mu = nn.Linear(HIDDEN_SIZE * 2, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_SIZE * 2, LATENT_DIM)
        
        # Decoder: Enhanced architecture
        self.decoder_lstm = nn.LSTM(
            LATENT_DIM + COND_DIM, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.1
        )
        
        # Output projection with multiple heads
        self.fc_out = nn.Linear(HIDDEN_SIZE, INPUT_DIM)
        self.residual_proj = nn.Linear(LATENT_DIM + COND_DIM, INPUT_DIM)
        
        # Additional refinement layers with residual connections
        self.refinement_layers = nn.ModuleList([
            ResidualBlock(INPUT_DIM, dropout=0.1)
            for _ in range(2)
        ])
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE, INPUT_DIM)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x, cond):
        c = cond.unsqueeze(1).repeat(1, x.size(1), 1)
        x_cond = torch.cat([x, c], dim=-1)
        
        # Bidirectional LSTM
        lstm_out, (h, _) = self.encoder_lstm(x_cond)
        
        # Apply transformer layers
        transformer_out = lstm_out
        for transformer in self.transformer_layers:
            transformer_out = transformer(transformer_out)
        
        # Apply residual blocks
        residual_out = transformer_out
        for residual in self.residual_blocks:
            residual_out = residual(residual_out)
        
        # Global average pooling with attention
        attention_weights = F.softmax(torch.sum(residual_out, dim=-1), dim=1)
        context = torch.sum(residual_out * attention_weights.unsqueeze(-1), dim=1)
        
        return self.fc_mu(context), self.fc_logvar(context)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        c = cond.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        z_seq = z.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        z_cond = torch.cat([z_seq, c], dim=-1)
        
        # LSTM decoder
        out, _ = self.decoder_lstm(z_cond)
        
        # Main output
        main_out = torch.tanh(self.fc_out(out))
        
        # Residual connection
        residual = torch.tanh(self.residual_proj(z_cond))
        
        # Combine outputs
        combined = main_out + 0.1 * residual
        
        # Apply refinement layers with residual connections
        refined = combined
        for refinement in self.refinement_layers:
            refined = refinement(refined)
        
        # Final output layer
        final_out = self.final_layer(refined)
        
        return torch.tanh(final_out)  # [-1,1]

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

# Keep the old classes for backward compatibility
class HighFidelityVAE(UltraHighFidelityVAE):
    pass

class VAE(UltraHighFidelityVAE):
    pass

def vae_loss(recon, x, mu, logvar, beta=0.001, alpha=0.01, gamma=0.05, delta=0.01, spatial_weight=0.05):
    """Ultra-high-fidelity VAE loss with spatial constraints"""
    # Reconstruction loss (MSE)
    mse_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence loss
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Smooth L1 loss for better gradient stability
    smooth_l1_loss = F.smooth_l1_loss(recon, x, reduction='mean')
    
    # Perceptual loss (L1)
    perceptual_loss = F.l1_loss(recon, x, reduction='mean')
    
    # Cosine similarity loss
    cos_loss = 1 - F.cosine_similarity(recon.view(recon.size(0), -1), 
                                      x.view(x.size(0), -1), dim=1).mean()
    
    # Spatial constraint losses
    spatial_losses = calculate_spatial_constraints(recon)
    
    # Combined loss
    total_loss = (mse_loss + alpha * smooth_l1_loss + beta * kld_loss + 
                  gamma * perceptual_loss + delta * cos_loss + 
                  spatial_weight * spatial_losses['total'])
    
    return total_loss, {
        'mse': mse_loss.item(),
        'smooth_l1': smooth_l1_loss.item(),
        'kld': kld_loss.item(),
        'perceptual': perceptual_loss.item(),
        'cosine': cos_loss.item(),
        'volume': spatial_losses['volume'].item(),
        'smoothness': spatial_losses['smoothness'].item(),
        'length': spatial_losses['length'].item(),
        'spatial_total': spatial_losses['total'].item(),
        'total': total_loss.item()
    }

def calculate_spatial_constraints(recon):
    """Calculate spatial constraint losses for realistic track generation."""
    batch_size, seq_len, dims = recon.shape
    
    # Volume constraint: ensure tracks stay within realistic bounds
    # Convert from [-1,1] back to real coordinates for constraint checking
    from neutron_vae.config import BOUNDS
    xyz_min = torch.tensor(BOUNDS['min'], device=recon.device, dtype=recon.dtype)
    xyz_max = torch.tensor(BOUNDS['max'], device=recon.device, dtype=recon.dtype)
    
    # Convert normalized coordinates back to real space
    recon_real = (recon + 1) / 2 * (xyz_max - xyz_min) + xyz_min
    
    # Volume constraint loss: penalize points outside bounds
    volume_loss = torch.mean(F.relu(recon_real - xyz_max) + F.relu(xyz_min - recon_real))
    
    # Track smoothness constraint
    if seq_len > 1:
        # Calculate second derivatives for smoothness
        second_deriv = recon[:, 2:] - 2 * recon[:, 1:-1] + recon[:, :-2]
        smoothness_loss = torch.mean(torch.norm(second_deriv, dim=-1))
    else:
        smoothness_loss = torch.tensor(0.0, device=recon.device)
    
    # Track length constraint
    if seq_len > 1:
        # Calculate track lengths
        diffs = recon[:, 1:] - recon[:, :-1]
        track_lengths = torch.sum(torch.norm(diffs, dim=-1), dim=1)
        
        # Penalize tracks that are too short or too long
        from neutron_vae.config import SPATIAL_CONSTRAINTS
        min_length = SPATIAL_CONSTRAINTS['min_track_length']
        max_length = SPATIAL_CONSTRAINTS['max_track_length']
        
        # Normalize lengths to real space
        length_scale = torch.norm(xyz_max - xyz_min)
        track_lengths_real = track_lengths * length_scale / 2  # Convert from normalized to real
        
        length_loss = torch.mean(
            F.relu(min_length - track_lengths_real) + 
            F.relu(track_lengths_real - max_length)
        )
    else:
        length_loss = torch.tensor(0.0, device=recon.device)
    
    # Physical constraint: ensure tracks don't have impossible jumps
    if seq_len > 1:
        # Calculate step sizes
        step_sizes = torch.norm(recon[:, 1:] - recon[:, :-1], dim=-1)
        # Penalize steps that are too large (unrealistic jumps)
        max_step = 0.5  # Maximum normalized step size
        step_loss = torch.mean(F.relu(step_sizes - max_step))
    else:
        step_loss = torch.tensor(0.0, device=recon.device)
    
    total_spatial_loss = volume_loss + smoothness_loss + length_loss + step_loss
    
    return {
        'volume': volume_loss,
        'smoothness': smoothness_loss,
        'length': length_loss,
        'step': step_loss,
        'total': total_spatial_loss
    }