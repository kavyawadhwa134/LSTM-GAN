import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMWGANGenerator(nn.Module):
    """
    LSTM-based Generator for WGAN with Event Prediction
    - Generates neutron tracking sequences
    - Predicts physics events (scattering, absorption, fission, leakage, capture)
    """
    
    def __init__(self, input_dim=7, hidden_dim=256, num_layers=3, sequence_length=50, 
                 num_events=5, device='cpu'):
        super(LSTMWGANGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_events = num_events
        self.device = device
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention for better sequence modeling
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Output heads for different features
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # x, y, z positions
            nn.Tanh()
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # vx, vy, vz velocities
            nn.Tanh()
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),  # energy
            nn.Sigmoid()
        )
        
        # Event prediction head
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_events),  # 5 events: scattering, absorption, fission, leakage, capture
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, noise):
        # Input projection
        x = self.input_projection(noise)
        
        # LSTM processing
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)
        
        # Generate outputs
        positions = self.position_head(x)
        velocities = self.velocity_head(x)
        energies = self.energy_head(x)
        events = self.event_head(x)
        
        # Combine outputs
        output = torch.cat([positions, velocities, energies], dim=-1)
        
        return output, events

class LSTMWGANDiscriminator(nn.Module):
    """
    LSTM-based Discriminator (Critic) for WGAN
    - No sigmoid output (raw scores)
    - Gradient penalty support
    """
    
    def __init__(self, input_dim=7, hidden_dim=256, num_layers=3, device='cpu'):
        super(LSTMWGANDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head (no sigmoid for WGAN)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Raw score, no activation
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # LSTM processing
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification (raw score)
        output = self.classifier(x)
        
        return output

class LSTMWGANWithEvents:
    """
    LSTM-WGAN with Event Prediction for Neutron Physics
    """
    
    def __init__(self, input_dim=7, hidden_dim=256, num_layers=3, sequence_length=50, 
                 num_events=5, device='cpu'):
        self.device = device
        self.sequence_length = sequence_length
        self.num_events = num_events
        
        # Initialize models
        self.generator = LSTMWGANGenerator(
            input_dim, hidden_dim, num_layers, sequence_length, num_events, device
        ).to(device)
        
        self.discriminator = LSTMWGANDiscriminator(
            input_dim, hidden_dim, num_layers, device
        ).to(device)
        
        # WGAN optimizers (RMSprop recommended)
        self.g_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=0.00005)
        self.d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00005)
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.gradient_penalties = []
        
        # WGAN parameters
        self.lambda_gp = 10.0  # Gradient penalty weight
        self.n_critic = 5      # Number of discriminator updates per generator update
        
        # Event names
        self.event_names = ['scattering', 'absorption', 'fission', 'leakage', 'capture']
    
    def compute_gradient_penalty(self, real_sequences, fake_sequences):
        """Compute gradient penalty for WGAN"""
        batch_size = real_sequences.size(0)
        
        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        interpolated = alpha * real_sequences + (1 - alpha) * fake_sequences
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_sequences):
        """Single training step for WGAN"""
        batch_size = real_sequences.size(0)
        
        # Train Discriminator (Critic) multiple times
        for _ in range(self.n_critic):
            self.d_optimizer.zero_grad()
            
            # Real sequences
            real_output = self.discriminator(real_sequences)
            d_real_loss = -torch.mean(real_output)
            
            # Fake sequences
            noise = torch.randn(batch_size, self.sequence_length, 7).to(self.device)
            fake_sequences, _ = self.generator(noise)
            fake_output = self.discriminator(fake_sequences.detach())
            d_fake_loss = torch.mean(fake_output)
            
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_sequences, fake_sequences)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gradient_penalty
            
            d_loss.backward()
            self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake sequences
        noise = torch.randn(batch_size, self.sequence_length, 7).to(self.device)
        fake_sequences, fake_events = self.generator(noise)
        fake_output = self.discriminator(fake_sequences)
        
        # Generator loss (Wasserstein loss)
        g_loss = -torch.mean(fake_output)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item(), gradient_penalty.item()
    
    def generate_sequences_with_events(self, num_sequences):
        """Generate synthetic sequences with event predictions"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_sequences, self.sequence_length, 7).to(self.device)
            generated_sequences, generated_events = self.generator(noise)
        self.generator.train()
        return generated_sequences.cpu().numpy(), generated_events.cpu().numpy()
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'gradient_penalties': self.gradient_penalties
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.g_losses = checkpoint['g_losses']
        self.d_losses = checkpoint['d_losses']
        self.gradient_penalties = checkpoint['gradient_penalties']
    
    def get_model_info(self):
        """Get model information"""
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        
        return {
            'generator_parameters': g_params,
            'discriminator_parameters': d_params,
            'total_parameters': g_params + d_params,
            'device': self.device,
            'input_dim': 7,
            'hidden_dim': 256,
            'sequence_length': 50,
            'num_layers': 3,
            'num_events': 5,
            'event_names': self.event_names
        }
