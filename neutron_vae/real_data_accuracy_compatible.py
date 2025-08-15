# real_data_accuracy_compatible.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, ks_2samp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neutron_vae.model import VAE  # Use the base VAE class
from neutron_vae.load_data import load_and_split_tracks, preprocess_tracks
from neutron_vae.config import *

class CompatibleRealDataAccuracyEvaluator:
    def __init__(self, model_path='neutron_vae_best.pth'):
        self.device = DEVICE
        
        # Load the checkpoint to get the actual model configuration
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine the model architecture from the checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            latent_dim = config.get('latent_dim', 32)
            hidden_size = config.get('hidden_size', 128)
            num_layers = config.get('num_layers', 2)
        else:
            # Default to the values that match your trained model
            latent_dim = 32
            hidden_size = 128
            num_layers = 2
        
        print(f"📊 Detected model configuration:")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Num layers: {num_layers}")
        
        # Create model with the correct architecture
        self.model = VAE().to(self.device)
        
        # Load model state
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {model_path}")
            if 'loss' in checkpoint:
                print(f"   Training loss: {checkpoint['loss']:.6f}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {model_path} (old format)")
            
        # Load real data
        raw_tracks = load_and_split_tracks()
        self.tracks_norm, self.conditions, self.xyz_min, self.xyz_max = preprocess_tracks(raw_tracks)
        self.tracks_tensor = torch.tensor(self.tracks_norm, dtype=torch.float32).to(self.device)
        self.conditions_tensor = torch.tensor(self.conditions, dtype=torch.float32).to(self.device)
        
        # Convert to [-1,1] range
        self.tracks_tensor = 2 * self.tracks_tensor - 1
        
        # Convert to real coordinates for comparison
        self.real_tracks = []
        for track in self.tracks_norm:
            track_real = track * (self.xyz_max - self.xyz_min) + self.xyz_min
            self.real_tracks.append(track_real)
        
    def generate_samples(self, n_samples=1000):
        """Generate samples from the model"""
        self.model.eval()
        generated_tracks = []
        
        with torch.no_grad():
            for i in range(n_samples):
                # Sample random latent vector
                z = torch.randn(1, 32, device=self.device)  # Use 32 for compatibility
                
                # Use random condition from training data
                cond_idx = np.random.randint(0, len(self.conditions))
                cond = self.conditions_tensor[cond_idx:cond_idx+1]
                
                # Generate track
                recon = self.model.decode(z, cond)
                
                # Convert to real coordinates
                track_norm = (recon[0].cpu().numpy() + 1) / 2  # [0,1]
                track_real = track_norm * (self.xyz_max - self.xyz_min) + self.xyz_min
                generated_tracks.append(track_real)
        
        return np.array(generated_tracks)
    
    def calculate_track_statistics(self, tracks):
        """Calculate statistics for a set of tracks"""
        lengths = []
        curvatures = []
        smoothness = []
        velocities = []
        
        for track in tracks:
            # Track length
            diffs = np.diff(track, axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
            lengths.append(length)
            
            # Average curvature
            if len(track) > 2:
                curvatures_track = []
                for j in range(1, len(track) - 1):
                    v1 = track[j] - track[j-1]
                    v2 = track[j+1] - track[j]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        curvatures_track.append(angle)
                if curvatures_track:
                    curvatures.append(np.mean(curvatures_track))
            
            # Smoothness (variance of second derivatives)
            if len(track) > 2:
                second_derivs = np.diff(track, n=2, axis=0)
                smoothness.append(np.var(second_derivs))
            
            # Average velocity
            if len(diffs) > 0:
                velocities.append(np.mean(np.linalg.norm(diffs, axis=1)))
        
        return {
            'lengths': np.array(lengths),
            'curvatures': np.array(curvatures),
            'smoothness': np.array(smoothness),
            'velocities': np.array(velocities)
        }
    
    def compare_distributions(self, real_stats, gen_stats):
        """Compare distributions using statistical tests"""
        comparisons = {}
        
        for stat_name in ['lengths', 'curvatures', 'smoothness', 'velocities']:
            if stat_name in real_stats and stat_name in gen_stats:
                real_data = real_stats[stat_name]
                gen_data = gen_stats[stat_name]
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = ks_2samp(real_data, gen_data)
                
                # Wasserstein distance
                wasserstein_dist = wasserstein_distance(real_data, gen_data)
                
                # Mean and std comparison
                real_mean, real_std = np.mean(real_data), np.std(real_data)
                gen_mean, gen_std = np.mean(gen_data), np.std(gen_data)
                
                comparisons[stat_name] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'wasserstein_distance': wasserstein_dist,
                    'real_mean': real_mean,
                    'real_std': real_std,
                    'gen_mean': gen_mean,
                    'gen_std': gen_std,
                    'mean_diff': abs(real_mean - gen_mean),
                    'std_diff': abs(real_std - gen_std)
                }
        
        return comparisons
    
    def calculate_spatial_accuracy(self, real_tracks, gen_tracks):
        """Calculate spatial accuracy metrics"""
        # Flatten all tracks for comparison
        real_points = np.vstack([track.flatten() for track in real_tracks])
        gen_points = np.vstack([track.flatten() for track in gen_tracks])
        
        # Calculate distances between real and generated distributions
        distances = cdist(real_points, gen_points)
        min_distances = np.min(distances, axis=1)
        
        return {
            'mean_min_distance': np.mean(min_distances),
            'std_min_distance': np.std(min_distances),
            'max_min_distance': np.max(min_distances),
            'coverage_score': np.mean(min_distances < 0.1)  # Points within 0.1 units
        }
    
    def evaluate_reconstruction_accuracy(self):
        """Evaluate how well the model reconstructs real data"""
        self.model.eval()
        with torch.no_grad():
            recon, _, _ = self.model(self.tracks_tensor, self.conditions_tensor)
            
        # Convert back to real coordinates
        original = (self.tracks_tensor + 1) / 2
        reconstructed = (recon + 1) / 2
        
        # Calculate per-track reconstruction error
        track_errors = []
        for i in range(len(self.real_tracks)):
            orig_track = original[i].cpu().numpy()
            recon_track = reconstructed[i].cpu().numpy()
            
            # Convert to real coordinates
            orig_real = orig_track * (self.xyz_max - self.xyz_min) + self.xyz_min
            recon_real = recon_track * (self.xyz_max - self.xyz_min) + self.xyz_min
            
            # Calculate point-wise distances
            distances = np.linalg.norm(orig_real - recon_real, axis=1)
            track_errors.append(np.mean(distances))
        
        return {
            'mean_track_error': np.mean(track_errors),
            'std_track_error': np.std(track_errors),
            'max_track_error': np.max(track_errors),
            'track_errors': np.array(track_errors)
        }
    
    def run_comprehensive_evaluation(self, n_samples=1000):
        """Run comprehensive evaluation against real data"""
        print("🧪 Running Comprehensive Real Data Accuracy Evaluation")
        print("=" * 70)
        
        # Generate samples
        print(f"\n🎲 Generating {n_samples} samples...")
        generated_tracks = self.generate_samples(n_samples)
        
        # Calculate statistics
        print("📊 Calculating track statistics...")
        real_stats = self.calculate_track_statistics(self.real_tracks)
        gen_stats = self.calculate_track_statistics(generated_tracks)
        
        # Compare distributions
        print("🔍 Comparing distributions...")
        comparisons = self.compare_distributions(real_stats, gen_stats)
        
        # Spatial accuracy
        print("📍 Calculating spatial accuracy...")
        spatial_accuracy = self.calculate_spatial_accuracy(self.real_tracks, generated_tracks)
        
        # Reconstruction accuracy
        print("🔄 Evaluating reconstruction accuracy...")
        reconstruction_accuracy = self.evaluate_reconstruction_accuracy()
        
        # Calculate overall accuracy score
        accuracy_score = self.calculate_overall_accuracy(comparisons, spatial_accuracy, reconstruction_accuracy)
        
        # Print results
        self.print_results(comparisons, spatial_accuracy, reconstruction_accuracy, accuracy_score)
        
        # Plot results
        self.plot_results(real_stats, gen_stats, comparisons)
        
        return {
            'comparisons': comparisons,
            'spatial_accuracy': spatial_accuracy,
            'reconstruction_accuracy': reconstruction_accuracy,
            'accuracy_score': accuracy_score,
            'real_stats': real_stats,
            'gen_stats': gen_stats
        }
    
    def calculate_overall_accuracy(self, comparisons, spatial_accuracy, reconstruction_accuracy):
        """Calculate overall accuracy score based on real data comparison"""
        score = 0
        max_score = 100
        
        # Distribution similarity (40 points)
        for stat_name, comp in comparisons.items():
            # KS test p-value (higher is better)
            ks_score = min(10, comp['ks_pvalue'] * 20)
            
            # Wasserstein distance (lower is better)
            wasserstein_score = max(0, 10 - comp['wasserstein_distance'] * 10)
            
            # Mean/std similarity
            mean_similarity = max(0, 10 - comp['mean_diff'] * 10)
            std_similarity = max(0, 10 - comp['std_diff'] * 10)
            
            score += (ks_score + wasserstein_score + mean_similarity + std_similarity) / 4
        
        # Spatial accuracy (30 points)
        coverage_score = spatial_accuracy['coverage_score'] * 15
        distance_score = max(0, 15 - spatial_accuracy['mean_min_distance'] * 50)
        score += coverage_score + distance_score
        
        # Reconstruction accuracy (30 points)
        recon_score = max(0, 30 - reconstruction_accuracy['mean_track_error'] * 100)
        score += recon_score
        
        return min(100, score)
    
    def print_results(self, comparisons, spatial_accuracy, reconstruction_accuracy, accuracy_score):
        """Print evaluation results"""
        print(f"\n📈 REAL DATA ACCURACY RESULTS")
        print("=" * 50)
        
        print(f"\n🎯 Overall Accuracy Score: {accuracy_score:.2f}/100")
        
        print(f"\n📊 Distribution Comparisons:")
        for stat_name, comp in comparisons.items():
            print(f"  {stat_name.upper()}:")
            print(f"    KS Test: p={comp['ks_pvalue']:.4f} (stat={comp['ks_statistic']:.4f})")
            print(f"    Wasserstein Distance: {comp['wasserstein_distance']:.4f}")
            print(f"    Mean: Real={comp['real_mean']:.4f}, Gen={comp['gen_mean']:.4f}")
            print(f"    Std: Real={comp['real_std']:.4f}, Gen={comp['gen_std']:.4f}")
        
        print(f"\n📍 Spatial Accuracy:")
        print(f"  Mean Min Distance: {spatial_accuracy['mean_min_distance']:.4f}")
        print(f"  Coverage Score: {spatial_accuracy['coverage_score']:.4f}")
        
        print(f"\n🔄 Reconstruction Accuracy:")
        print(f"  Mean Track Error: {reconstruction_accuracy['mean_track_error']:.4f}")
        print(f"  Std Track Error: {reconstruction_accuracy['std_track_error']:.4f}")
    
    def plot_results(self, real_stats, gen_stats, comparisons):
        """Plot comparison results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot distributions
        stat_names = ['lengths', 'curvatures', 'smoothness']
        for i, stat_name in enumerate(stat_names):
            if stat_name in real_stats and stat_name in gen_stats:
                axes[0, i].hist(real_stats[stat_name], bins=30, alpha=0.7, label='Real', density=True)
                axes[0, i].hist(gen_stats[stat_name], bins=30, alpha=0.7, label='Generated', density=True)
                axes[0, i].set_title(f'{stat_name.capitalize()} Distribution')
                axes[0, i].legend()
                axes[0, i].grid(True)
        
        # Plot KS test results
        ks_stats = [comp['ks_statistic'] for comp in comparisons.values()]
        ks_pvalues = [comp['ks_pvalue'] for comp in comparisons.values()]
        stat_labels = list(comparisons.keys())
        
        axes[1, 0].bar(stat_labels, ks_stats)
        axes[1, 0].set_title('KS Test Statistics')
        axes[1, 0].set_ylabel('KS Statistic')
        
        axes[1, 1].bar(stat_labels, ks_pvalues)
        axes[1, 1].set_title('KS Test P-values')
        axes[1, 1].set_ylabel('P-value')
        axes[1, 1].axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
        axes[1, 1].legend()
        
        # Plot Wasserstein distances
        wasserstein_dists = [comp['wasserstein_distance'] for comp in comparisons.values()]
        axes[1, 2].bar(stat_labels, wasserstein_dists)
        axes[1, 2].set_title('Wasserstein Distances')
        axes[1, 2].set_ylabel('Distance')
        
        plt.tight_layout()
        plt.savefig('real_data_accuracy_results.png', dpi=150, bbox_inches='tight')
        print("✅ Real data accuracy results saved as 'real_data_accuracy_results.png'")

if __name__ == "__main__":
    evaluator = CompatibleRealDataAccuracyEvaluator()
    results = evaluator.run_comprehensive_evaluation()
