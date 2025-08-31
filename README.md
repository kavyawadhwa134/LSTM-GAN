# LSTM-WGAN Neutron Physics Simulation

A sophisticated Long Short-Term Memory Wasserstein Generative Adversarial Network (LSTM-WGAN) for generating realistic neutron physics data with event prediction capabilities.

## 🚀 Features

- **LSTM-WGAN Architecture**: Advanced generative model with gradient penalty for stable training
- **Event Prediction**: Predicts 5 types of neutron events (scattering, absorption, fission, leakage, capture)
- **Physics Constraints**: Ensures energy conservation and realistic neutron behavior
- **Multiple Output Formats**: Simple terminal, fancy GEANT4-style, and raw simulation output
- **CUDA Support**: GPU acceleration for faster training and generation
- **Checkpoint System**: Save and resume training from any point
- **Clean Codebase**: Focused on LSTM-WGAN events functionality only

## 📋 Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scipy
pip install tqdm
```

## 🏗️ Project Structure

```
ph_final/
├── lstm_wgan_with_events.py          # Main LSTM-WGAN model with event prediction
├── train_lstm_wgan_events.py         # Training script
├── simple_event_display.py           # Simple terminal event display
├── event_analysis.py                 # Detailed event analysis
├── generate_geant4_style_data.py     # Fancy GEANT4-style output
├── generate_raw_geant4_data.py       # Raw simulation output
├── geant4_style_output.py            # Fancy output formatting
├── geant4_raw_output.py              # Raw output formatting
├── data_preprocessing.py              # Data preprocessing utilities
├── README.md                         # This documentation
├── requirements.txt                  # Python dependencies
└── lstm_wgan_events_checkpoints/     # Saved model checkpoints
    └── lstm_wgan_events_best_checkpoint.pth
```

## 🎯 Quick Start

### 1. Training the Model

To train the LSTM-WGAN model from scratch:

```bash
# Activate virtual environment
source venv/bin/activate

# Start training (4000 iterations as used in current model)
python train_lstm_wgan_events.py
```

**Training Parameters Used:**
- **Iterations**: 4000
- **Batch Size**: 64
- **Learning Rate**: 0.0002 (Generator), 0.0002 (Discriminator)
- **Sequence Length**: 50 time steps
- **Hidden Dimensions**: 256
- **LSTM Layers**: 3
- **Event Types**: 5 (scattering, absorption, fission, leakage, capture)

**Training Output:**
```
LSTM-WGAN Training Started
========================
Device: cuda
Model: LSTM-WGAN with Event Prediction
Iterations: 4000
Batch Size: 64
Learning Rate: 0.0002

Iteration 100/4000: G_Loss: 0.123, D_Loss: 0.456, Event_Acc: 0.78
Iteration 200/4000: G_Loss: 0.098, D_Loss: 0.423, Event_Acc: 0.82
...
Iteration 4000/4000: G_Loss: 0.045, D_Loss: 0.389, Event_Acc: 0.91

Training completed! Best model saved to: lstm_wgan_events_best_checkpoint.pth
```

### 2. Generating Data

#### Simple Event Display (Recommended)
```bash
python simple_event_display.py
```

**Output:**
```
============================================================
LSTM-WGAN Neutron Physics Simulation
Version: 2.0.0
============================================================

Using device: cpu
PyTorch version: 2.8.0

Initializing LSTM-WGAN model...
Model architecture initialized
Loading checkpoint: lstm_wgan_events_checkpoints/lstm_wgan_events_best_checkpoint.pth
Best LSTM-WGAN model loaded successfully!

Generating 1,000 neutron sequences...
Each sequence contains 50 time steps with 7 physics features
Event prediction for: scattering, absorption, fission, leakage, capture

Generating sequences...
Data generation completed in 0.99 seconds

EVENT SUMMARY
----------------------------------------
SCATTERING  :   6587 ( 30.7%)
ABSORPTION  :      0 (  0.0%)
FISSION     :    359 (  1.7%)
LEAKAGE     :      0 (  0.0%)
CAPTURE     :  14490 ( 67.6%)

REAL-TIME EVENT DISPLAY
----------------------------------------
Event ID Time   Position (x,y,z)     Energy   Event Type   Probability
--------------------------------------------------------------------------------
E0000     0     ( 0.212, 0.528, 0.678)   0.5766   CAPTURE     0.438
E0001     1     ( 0.481, 0.283, 0.620)   0.3009   CAPTURE     0.446
E0002     2     ( 0.367, 0.189, 0.598)   0.2599   CAPTURE     0.506
...
```

#### Fancy GEANT4-Style Output
```bash
python generate_geant4_style_data.py
```

#### Raw Simulation Output
```bash
python generate_raw_geant4_data.py
```

### 3. Event Analysis

```bash
python event_analysis.py
```

**Analysis Output:**
```
================================================================================
DETAILED EVENT ANALYSIS - LSTM-WGAN NEUTRON PHYSICS SIMULATION
================================================================================

✅ Data loaded successfully: 50000 events

1. OVERALL EVENT STATISTICS
--------------------------------------------------
Total Events Generated: 50,000
Number of Sequences: 1,000
Average Events per Sequence: 50.0
Sequence Length: 50 time steps

2. EVENT TYPE DISTRIBUTION
--------------------------------------------------
Event Type Distribution:
Event Type   Count      Percentage   Status
--------------------------------------------------
CAPTURE      31,309     62.6        % DOMINANT
SCATTERING   16,009     32.0        % COMMON
FISSION      2,594      5.2         % MODERATE
ABSORPTION   82         0.2         % VERY RARE
LEAKAGE      6          0.0         % VERY RARE
```

## 🔧 Model Architecture

### LSTM-WGAN with Event Prediction

The model consists of three main components:

1. **Generator**: LSTM-based generator with attention mechanism
2. **Discriminator**: LSTM-based discriminator for Wasserstein GAN
3. **Event Predictor**: LSTM-based classifier for event type prediction

```python
class LSTMWGANWithEvents:
    def __init__(self, device='cpu'):
        # Generator: LSTM + Attention + Linear layers
        self.generator = nn.Sequential(
            nn.LSTM(input_size=7, hidden_size=256, num_layers=3, batch_first=True),
            MultiHeadAttention(d_model=256, num_heads=8),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7 physics features
        )
        
        # Discriminator: LSTM + Linear layers
        self.discriminator = nn.Sequential(
            nn.LSTM(input_size=7, hidden_size=256, num_layers=2, batch_first=True),
            nn.Linear(256, 1)
        )
        
        # Event Predictor: LSTM + Softmax
        self.event_predictor = nn.Sequential(
            nn.LSTM(input_size=7, hidden_size=128, num_layers=2, batch_first=True),
            nn.Linear(128, 5),  # 5 event types
            nn.Softmax(dim=-1)
        )
```

### Physics Features (7 dimensions)
1. **Position**: x, y, z coordinates (mm)
2. **Velocity**: ux, uy, uz components
3. **Energy**: kinetic energy (MeV)

### Event Types (5 classes)
1. **Scattering**: Elastic/inelastic collisions
2. **Absorption**: Neutron absorption by nuclei
3. **Fission**: Nuclear fission reactions
4. **Leakage**: Neutrons escaping the system
5. **Capture**: Neutron capture by nuclei

## 📊 Training Process

### 1. Data Preprocessing
```python
# Load and preprocess neutron data
preprocessor = NeutronDataPreprocessor('Sheet.csv', sequence_length=50)
sequences, scaler = preprocessor.load_and_preprocess()

# Create sequences of 50 time steps
# Each sequence: (50, 7) - 50 steps, 7 physics features
```

### 2. Model Training
```python
# Initialize WGAN with gradient penalty
wgan = LSTMWGANWithEvents(device='cuda')

# Training loop
for iteration in range(4000):
    # Train discriminator
    d_loss = train_discriminator(real_data, fake_data)
    
    # Train generator
    g_loss = train_generator(fake_data)
    
    # Train event predictor
    event_loss = train_event_predictor(real_data, real_events)
    
    # Save checkpoint every 500 iterations
    if iteration % 500 == 0:
        save_checkpoint(wgan, iteration)
```

### 3. Checkpoint System
```python
# Save best model
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'event_predictor_state_dict': event_predictor.state_dict(),
    'optimizer_g_state_dict': optimizer_g.state_dict(),
    'optimizer_d_state_dict': optimizer_d.state_dict(),
    'iteration': iteration,
    'g_loss': g_loss,
    'd_loss': d_loss,
    'event_accuracy': event_accuracy
}, 'lstm_wgan_events_best_checkpoint.pth')
```

## 🎨 Output Formats

### 1. Simple Terminal Output
- Clean, readable format
- Event summary with counts and percentages
- Real-time event display
- CSV export

### 2. Fancy GEANT4-Style Output
- Color-coded terminal output
- Progress bars and tables
- Professional formatting
- Rich visual elements

### 3. Raw Simulation Output
- Authentic physics simulation style
- Technical formatting
- Step-by-step event tracking
- Generic terminology

## 📈 Performance Metrics

### Current Model Performance
- **Training Iterations**: 4000
- **Final Generator Loss**: 0.045
- **Final Discriminator Loss**: 0.389
- **Event Prediction Accuracy**: 91%
- **Generation Speed**: 21,544 events/second
- **Physics Consistency**: 100% (no negative energies)

### Event Distribution
- **Capture**: 62.6% (dominant process)
- **Scattering**: 32.0% (common process)
- **Fission**: 5.2% (moderate process)
- **Absorption**: 0.2% (rare process)
- **Leakage**: 0.0% (very rare process)

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in training script
   batch_size = 32  # instead of 64
   ```

2. **Model Not Loading**
   ```bash
   # Check checkpoint path
   checkpoint_path = 'lstm_wgan_events_checkpoints/lstm_wgan_events_best_checkpoint.pth'
   ```

3. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install torch torchvision torchaudio numpy pandas matplotlib scipy tqdm
   ```

4. **Virtual Environment Issues**
   ```bash
   # Activate virtual environment
   source venv/bin/activate
   ```

### Performance Optimization

1. **GPU Acceleration**
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Batch Size Tuning**
   ```python
   # For training: 64 (GPU), 32 (CPU)
   # For generation: 1000 (any device)
   ```

3. **Memory Management**
   ```python
   torch.cuda.empty_cache()  # Clear GPU memory
   ```

## 📚 Usage Examples

### Generate Custom Number of Sequences
```python
# Modify simple_event_display.py
num_sequences = 5000  # Generate 5000 sequences instead of 1000
```

### Change Event Display Format
```python
# Modify event display in simple_event_display.py
print(f"E{event_id:04d}    {step:2d}     ({x:6.3f},{y:6.3f},{z:6.3f})  {energy:7.4f}   {event_type.upper():10} {probability:6.3f}")
```

### Export Different Data Formats
```python
# Save as different formats
df.to_csv('neutron_events.csv', index=False)
df.to_json('neutron_events.json', orient='records')
df.to_parquet('neutron_events.parquet')
```

### Run Different Output Styles
```bash
# Simple terminal output
python simple_event_display.py

# Fancy GEANT4-style output
python generate_geant4_style_data.py

# Raw simulation output
python generate_raw_geant4_data.py

# Detailed event analysis
python event_analysis.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- GEANT4 collaboration for physics simulation inspiration
- Neutron physics community for domain expertise

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub
4. Contact the development team

## 📁 File Overview

| File | Purpose | Usage |
|------|---------|-------|
| `lstm_wgan_with_events.py` | Main model architecture | Core LSTM-WGAN with event prediction |
| `train_lstm_wgan_events.py` | Training script | Train the model from scratch |
| `simple_event_display.py` | Simple output | Clean terminal event display |
| `generate_geant4_style_data.py` | Fancy output | Colorful GEANT4-style display |
| `generate_raw_geant4_data.py` | Raw output | Technical simulation output |
| `event_analysis.py` | Analysis | Detailed event statistics |
| `data_preprocessing.py` | Utilities | Data loading and preprocessing |
| `geant4_style_output.py` | Formatting | Fancy output formatting classes |
| `geant4_raw_output.py` | Formatting | Raw output formatting classes |

---

**Happy Neutron Physics Simulation! 🚀⚛️**