# accuracy_test.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import HighFidelityVAE
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
from neutron_vae.config import *

class HighFidelityAccuracyTester:
    def __init__(self, model_path='neutron_vae.pth'):
        self.device = DEVICE
        self.model = HighFidelityVAE().to(self.device)
        
        # Load model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model from {model_path}")
                if 'loss' in checkpoint:
                    print(f"   Training loss: {checkpoint['loss']:.6f}")
            else:
                # Old format
                self.model.load_state_dict(checkpoint)
                print(f"✅ Loaded model from {model_path} (old format)")
        else:
            print(f"❌ Model file {model_path} not found")
            return
            
        # Load data
        raw_tracks = load_and_split_tracks()
        self.tracks_norm, self.conditions, self.xyz_min, self.xyz_max = preprocess_tracks(raw_tracks)
        self.tracks_tensor = torch.tensor(self.tracks_norm, dtype=torch.float32).to(self.device)
        self.conditions_tensor = torch.tensor(self.conditions, dtype=torch.float32).to(self.device)
        
        # Convert to [-1,1] range
        self.tracks_tensor = 2 * self.tracks_tensor - 1
        
    def reconstruction_accuracy(self):
        """Test reconstruction accuracy with high-fidelity metrics"""
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(self.tracks_tensor, self.conditions_tensor)
            
        # Convert back to [0,1] range
        original = (self.tracks_tensor + 1) / 2
        reconstructed = (recon + 1) / 2
        
        # Calculate metrics
        mse = mean_squared_error(original.cpu().numpy().flatten(), 
                               reconstructed.cpu().numpy().flatten())
        mae = mean_absolute_error(original.cpu().numpy().flatten(), 
                                reconstructed.cpu().numpy().flatten())
        
        # Per-track metrics
        track_mse = []
        track_mae = []
        track_cosine_similarity = []
        
        for i in range(len(self.tracks_norm)):
            orig_track = original[i].cpu().numpy()
            recon_track = reconstructed[i].cpu().numpy()
            
            # MSE and MAE
            track_mse.append(mean_squared_error(orig_track.flatten(), recon_track.flatten()))
            track_mae.append(mean_absolute_error(orig_track.flatten(), recon_track.flatten()))
            
            # Cosine similarity
            orig_flat = orig_track.flatten()
            recon_flat = recon_track.flatten()
            cos_sim = np.dot(orig_flat, recon_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(recon_flat))
            track_cosine_similarity.append(cos_sim)
        
        return {
            'overall_mse': mse,
            'overall_mae': mae,
            'track_mse': np.array(track_mse),
            'track_mae': np.array(track_mae),
            'track_cosine_similarity': np.array(track_cosine_similarity),
            'mean_track_mse': np.mean(track_mse),
            'mean_track_mae': np.mean(track_mae),
            'mean_cosine_similarity': np.mean(track_cosine_similarity),
            'std_track_mse': np.std(track_mse),
            'std_track_mae': np.std(track_mae),
            'std_cosine_similarity': np.std(track_cosine_similarity)
        }
    
    def latent_space_analysis(self):
        """Analyze latent space properties"""
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(self.tracks_tensor, self.conditions_tensor)
            z = self.model.reparameterize(mu, logvar)
            
        # Calculate KL divergence
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Latent space statistics
        z_np = z.cpu().numpy()
        mu_np = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()
        
        # Calculate latent space coverage
        latent_coverage = np.std(z_np, axis=0).mean()
        
        return {
            'kl_divergence': kld.item(),
            'latent_mean': np.mean(z_np, axis=0),
            'latent_std': np.std(z_np, axis=0),
            'latent_min': np.min(z_np, axis=0),
            'latent_max': np.max(z_np, axis=0),
            'latent_coverage': latent_coverage,
            'mu_mean': np.mean(mu_np, axis=0),
            'logvar_mean': np.mean(logvar_np, axis=0)
        }
    
    def generation_quality(self, n_samples=200):
        """Test generation quality by sampling from latent space"""
        self.model.eval()
        generated_tracks = []
        
        with torch.no_grad():
            for i in range(n_samples):
                # Sample random latent vector
                z = torch.randn(1, LATENT_DIM, device=self.device)
                
                # Use random condition from training data
                cond_idx = np.random.randint(0, len(self.conditions))
                cond = self.conditions_tensor[cond_idx:cond_idx+1]
                
                # Generate track
                recon = self.model.decode(z, cond)
                generated_tracks.append(recon.cpu().numpy())
        
        generated_tracks = np.array(generated_tracks)
        
        # Calculate statistics of generated tracks
        track_lengths = []
        track_curvatures = []
        track_smoothness = []
        
        for track in generated_tracks:
            # Convert to real coordinates
            track_real = (track[0] + 1) / 2  # [0,1]
            track_real = track_real * (self.xyz_max - self.xyz_min) + self.xyz_min
            
            # Calculate track length
            diffs = np.diff(track_real, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
            track_lengths.append(length)
            
            # Calculate average curvature
            if len(track_real) > 2:
                curvatures = []
                for j in range(1, len(track_real) - 1):
                    v1 = track_real[j] - track_real[j-1]
                    v2 = track_real[j+1] - track_real[j]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        curvatures.append(angle)
                if curvatures:
                    track_curvatures.append(np.mean(curvatures))
            
            # Calculate smoothness (variance of second derivatives)
            if len(track_real) > 2:
                second_derivs = np.diff(track_real, n=2, axis=0)
                smoothness = np.var(second_derivs)
                track_smoothness.append(smoothness)
        
        return {
            'generated_tracks': generated_tracks,
            'mean_track_length': np.mean(track_lengths),
            'std_track_length': np.std(track_lengths),
            'mean_curvature': np.mean(track_curvatures) if track_curvatures else 0,
            'std_curvature': np.std(track_curvatures) if track_curvatures else 0,
            'mean_smoothness': np.mean(track_smoothness) if track_smoothness else 0,
            'std_smoothness': np.std(track_smoothness) if track_smoothness else 0
        }
    
    def run_comprehensive_test(self):
        """Run all accuracy tests"""
        print("🧪 Running High-Fidelity Accuracy Tests...")
        print("=" * 60)
        
        # Reconstruction accuracy
        print("\n📊 Testing reconstruction accuracy...")
        recon_metrics = self.reconstruction_accuracy()
        print(f"Overall MSE: {recon_metrics['overall_mse']:.6f}")
        print(f"Overall MAE: {recon_metrics['overall_mae']:.6f}")
        print(f"Mean Cosine Similarity: {recon_metrics['mean_cosine_similarity']:.6f} ± {recon_metrics['std_cosine_similarity']:.6f}")
        print(f"Mean track MSE: {recon_metrics['mean_track_mse']:.6f} ± {recon_metrics['std_track_mse']:.6f}")
        print(f"Mean track MAE: {recon_metrics['mean_track_mae']:.6f} ± {recon_metrics['std_track_mae']:.6f}")
        
        # Latent space analysis
        print("\n🔍 Analyzing latent space...")
        latent_metrics = self.latent_space_analysis()
        print(f"KL Divergence: {latent_metrics['kl_divergence']:.6f}")
        print(f"Latent space coverage: {latent_metrics['latent_coverage']:.6f}")
        print(f"Latent space mean: {np.mean(latent_metrics['latent_mean']):.6f}")
        print(f"Latent space std: {np.mean(latent_metrics['latent_std']):.6f}")
        
        # Generation quality
        print("\n🎲 Testing generation quality...")
        gen_metrics = self.generation_quality()
        print(f"Mean generated track length: {gen_metrics['mean_track_length']:.6f} ± {gen_metrics['std_track_length']:.6f}")
        print(f"Mean curvature: {gen_metrics['mean_curvature']:.6f} ± {gen_metrics['std_curvature']:.6f}")
        print(f"Mean smoothness: {gen_metrics['mean_smoothness']:.6f} ± {gen_metrics['std_smoothness']:.6f}")
        
        # Overall accuracy score
        accuracy_score = self.calculate_accuracy_score(recon_metrics, latent_metrics, gen_metrics)
        print(f"\n🎯 Overall Accuracy Score: {accuracy_score:.2f}/100")
        
        return {
            'reconstruction': recon_metrics,
            'latent_space': latent_metrics,
            'generation': gen_metrics,
            'overall_score': accuracy_score
        }
    
    def calculate_accuracy_score(self, recon_metrics, latent_metrics, gen_metrics):
        """Calculate overall accuracy score (0-100) with high-fidelity criteria"""
        # Reconstruction score (40 points)
        recon_score = max(0, 40 * (1 - recon_metrics['overall_mse'] * 20))
        
        # Cosine similarity bonus (10 points)
        cosine_score = max(0, 10 * recon_metrics['mean_cosine_similarity'])
        
        # Latent space score (25 points)
        # Good KL divergence should be around 0.01-0.1 for high fidelity
        kl_score = max(0, 25 * (1 - abs(latent_metrics['kl_divergence'] - 0.05) * 10))
        
        # Latent coverage bonus (5 points)
        coverage_score = max(0, 5 * min(1.0, latent_metrics['latent_coverage']))
        
        # Generation score (20 points)
        # Check if generated tracks have reasonable properties
        length_score = max(0, 10 * (1 - abs(gen_metrics['mean_track_length'] - 15) / 15))
        smoothness_score = max(0, 10 * (1 - gen_metrics['mean_smoothness'] / 10))
        
        total_score = recon_score + cosine_score + kl_score + coverage_score + length_score + smoothness_score
        return min(100, total_score)
    
    def plot_results(self, results):
        """Plot accuracy test results"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Reconstruction error distribution
        axes[0, 0].hist(results['reconstruction']['track_mse'], bins=10, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Track MSE Distribution')
        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Frequency')
        
        # Cosine similarity distribution
        axes[0, 1].hist(results['reconstruction']['track_cosine_similarity'], bins=10, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Cosine Similarity Distribution')
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Frequency')
        
        # Latent space visualization
        axes[0, 2].scatter(results['latent_space']['latent_mean'][:8], 
                          results['latent_space']['latent_std'][:8], alpha=0.7)
        axes[0, 2].set_title('Latent Space (First 8 Dimensions)')
        axes[0, 2].set_xlabel('Mean')
        axes[0, 2].set_ylabel('Std')
        
        # Track length distribution
        if 'generated_tracks' in results['generation']:
            track_lengths = []
            for track in results['generation']['generated_tracks']:
                track_real = (track[0] + 1) / 2
                track_real = track_real * (self.xyz_max - self.xyz_min) + self.xyz_min
                diffs = np.diff(track_real, axis=0)
                length = np.sum(np.linalg.norm(diffs, axis=1))
                track_lengths.append(length)
            
            axes[0, 3].hist(track_lengths, bins=15, alpha=0.7, color='orange')
            axes[0, 3].set_title('Generated Track Lengths')
            axes[0, 3].set_xlabel('Length')
            axes[0, 3].set_ylabel('Frequency')
        
        # Accuracy score breakdown
        recon_score = max(0, 40 * (1 - results['reconstruction']['overall_mse'] * 20))
        cosine_score = max(0, 10 * results['reconstruction']['mean_cosine_similarity'])
        kl_score = max(0, 25 * (1 - abs(results['latent_space']['kl_divergence'] - 0.05) * 10))
        coverage_score = max(0, 5 * min(1.0, results['latent_space']['latent_coverage']))
        length_score = max(0, 10 * (1 - abs(results['generation']['mean_track_length'] - 15) / 15))
        smoothness_score = max(0, 10 * (1 - results['generation']['mean_smoothness'] / 10))
        
        scores = [recon_score, cosine_score, kl_score, coverage_score, length_score, smoothness_score]
        labels = ['Reconstruction', 'Cosine Sim', 'KL Div', 'Coverage', 'Length', 'Smoothness']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#99ccff']
        
        axes[1, 0].pie(scores, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1, 0].set_title('Accuracy Score Breakdown')
        
        # Metrics summary
        metrics_text = f"""
        Overall Score: {results['overall_score']:.1f}/100
        
        Reconstruction:
        - MSE: {results['reconstruction']['overall_mse']:.6f}
        - MAE: {results['reconstruction']['overall_mae']:.6f}
        - Cosine Sim: {results['reconstruction']['mean_cosine_similarity']:.4f}
        
        Latent Space:
        - KL Divergence: {results['latent_space']['kl_divergence']:.6f}
        - Coverage: {results['latent_space']['latent_coverage']:.4f}
        
        Generation:
        - Mean Length: {results['generation']['mean_track_length']:.2f}
        - Mean Curvature: {results['generation']['mean_curvature']:.4f}
        - Mean Smoothness: {results['generation']['mean_smoothness']:.4f}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=9, verticalalignment='center')
        axes[1, 1].set_title('Metrics Summary')
        axes[1, 1].axis('off')
        
        # Overall score bar
        axes[1, 2].bar(['Accuracy Score'], [results['overall_score']], 
                      color='lightblue', alpha=0.7)
        axes[1, 2].set_ylim(0, 100)
        axes[1, 2].set_title('Overall Accuracy Score')
        axes[1, 2].set_ylabel('Score')
        
        # Training progress (if available)
        if os.path.exists('training_progress.png'):
            img = plt.imread('training_progress.png')
            axes[1, 3].imshow(img)
            axes[1, 3].set_title('Training Progress')
            axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('high_fidelity_accuracy_results.png', dpi=150, bbox_inches='tight')
        print("✅ High-fidelity accuracy test results saved as 'high_fidelity_accuracy_results.png'")

# Backward compatibility
class AccuracyTester(HighFidelityAccuracyTester):
    pass

if __name__ == "__main__":
    tester = HighFidelityAccuracyTester()
    results = tester.run_comprehensive_test()
    tester.plot_results(results)
