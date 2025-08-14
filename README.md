## Neutron VAE (tabular)

Minimal Variational Autoencoder pipeline for tabular CSV data.

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Project layout

```
neutron_vae/
  config.py
  load_data.py
  model.py
  train.py
  generate.py
  visualize.py
  Sheet.csv
```

### Usage

1) Put your numeric tabular data in `neutron_vae/Sheet.csv` (headers required).

2) Train the VAE:

```bash
python -m neutron_vae.train
```

3) Generate samples from the trained model:

```bash
python -m neutron_vae.generate
```

4) Visualize generated feature histograms:

```bash
python -m neutron_vae.visualize
```

Configuration defaults live in `neutron_vae/config.py` and auto-detect `cpu/cuda/mps`.


