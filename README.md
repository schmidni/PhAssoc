# Phase Association for Seismic/Acoustic Emission Monitoring

## The Problem

When monitoring seismic or acoustic activity (e.g., in underground laboratories, mines, or natural earthquake regions), sensors detect wave arrivals as **picks**—timestamps marking when a seismic phase (P-wave or S-wave) reaches a station. The challenge is **phase association**: determining which picks belong to the same physical event.

This is particularly difficult when:
- Multiple events occur simultaneously or in quick succession
- Background noise creates false picks
- Not all stations detect every event
- Picks arrive out of temporal order

Accurate phase association is critical for locating events and understanding seismicity patterns.

## Repository Overview

This repository implements and evaluates multiple phase association algorithms, with a focus on acoustic emission data from underground laboratories. It contains:

1. **Baseline Methods**: Implementations of established algorithms
   - **GaMMA** (Gaussian Mixture Model-based association)
   - **PyOcto** (Oct-tree spatial partitioning)
   - **HARPA** (hybrid method)

2. **PADME**: A novel deep metric learning approach using transformer networks (see below)

3. **Synthetic Data Generation**: Tools to create realistic test catalogs with controlled parameters

4. **Evaluation Framework**: Metrics and comparison tools for systematic performance assessment

5. **Real-World Applications**: Experiments on datasets from Bedretto Lab and Ridgecrest

## PADME: Deep Metric Learning for Phase Association

**PADME** (Phase Association via Deep Metric Embeddings) is a novel approach that learns to cluster picks by embedding them in a metric space where picks from the same event are close together.

### Architecture

PADME uses a **Transformer-based neural network** (`PhasePickTransformer`) that:
- Takes pick features as input: time, location, amplitude, phase type, station ID
- Learns embeddings via **metric learning** objectives (triplet loss, contrastive loss, circle loss, etc.)
- Projects each pick into a low-dimensional embedding space (typically 16D)

The model architecture includes:
- Station embeddings (learnable per-station representations)
- Positional encodings (temporal sequence information)
- Multi-head self-attention layers
- MLP projection heads for final embeddings

### Two-Stage Clustering

PADME operates in two stages:

1. **Coarse Clustering**: Apply DBSCAN in time-space to create "slides"—rough temporal windows that likely contain picks from a manageable number of events

2. **Fine Clustering**: For each slide:
   - Pass picks through the trained transformer to generate embeddings
   - Apply DBSCAN in embedding space to identify event clusters
   - Automatically estimate DBSCAN epsilon using the elbow method

This hierarchical approach makes PADME scalable while maintaining high accuracy.

### Training

Training uses **metric learning losses** from the `pytorch-metric-learning` library:
- **Triplet Loss**: Learn to separate anchor-positive pairs from anchor-negative pairs
- **Multi-Similarity Loss**: Adaptive weighting of positive/negative pairs
- **Circle Loss**: Flexible margin-based learning
- **Contrastive Loss**: Pairwise similarity optimization

Hyperparameters (margin, alpha, beta) are scheduled to evolve during training for better convergence.

## Guide to the Repository

### Core Source Code (`src/`)

| Module | Purpose |
|--------|---------|
| `models.py` | Neural network architectures: `PhasePickTransformer`, `PhasePickMLP` |
| `losses.py` | Metric learning loss functions with scheduling |
| `dataset.py` | PyTorch Dataset classes for loading picks and catalogs |
| `runners.py` | Wrapper functions: `run_gamma()`, `run_pyocto()`, `run_harpa()`, `run_phassoc()` |
| `evaluate.py` | Evaluation pipelines for systematic benchmarking |
| `metrics.py` | `ClusterStatistics` class for computing precision, recall, ARI |
| `__init__.py` | PADME implementation: `associate_phassoc()`, `DBSCAN_cluster()` |

### Specialized Modules

**`src/synthetics/`**: Synthetic data generation
- `create_synthetic_catalog.py`: Generate random event catalogs
- `create_associations.py`: Simulate picks with travel times and noise
- `generate_poisson.py`: Poisson process event timing
- `simulate_magnitudes.py`: Magnitude distributions

**`src/gamma/`**: Modified GaMMA implementation
- Adapted from scikit-learn's Gaussian Mixture Models
- `_gaussian_mixture.py`, `_bayesian_mixture.py`: Core clustering
- `utils.py`: Association logic

**`src/plotting/`**: Visualization tools
- `arrivals.py`: Plot pick patterns
- `embeddings.py`: Visualize embedding spaces

### Experiments (`experiments/`)

| Folder | Description |
|--------|-------------|
| `base_models/` | Scripts to run GaMMA, PyOcto, HARPA, PADME individually |
| `comparisons/` | Systematic comparisons across methods and parameters |
| `m0/` | Analysis of Bedretto Lab magnitude-0 events |
| `ridgecrest/` | Ridgecrest earthquake sequence analysis |
| `playground/` | Exploratory notebooks and tests |

### Top-Level Scripts

- **`create_synthetic_data.py`**: Main script for generating test datasets
  - Configure parameters: event rate, duration, noise levels, station geometry
  - Outputs: `arrivals_*.csv` (picks) and `catalog_*.csv` (ground truth)

- **`create_synthetic_data_parallel.py`**: Parallelized version for large-scale generation

- **`metric.py`**: Standalone metric computation utilities

- **`confusion_factor.py`**: Analysis of event confusion patterns

### Data Organization

```
data/               # Generated synthetic catalogs
models/             # Trained PADME model weights
stations/           # Station coordinate files
reports/            # Analysis reports and figures
  ├── m0_report/       # Bedretto M0 analysis
  ├── comparison/      # Model comparisons
  └── project_report/  # Main report (LaTeX)
plots/              # Generated visualizations
```

