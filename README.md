<div align="center">

# Quirk Tracking Pipeline

A machine-learning track reconstruction pipeline for long-lived exotic particles ("quirks") at the LHC, built on the [ExaTrkX](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX) framework.

</div>

## Overview

This pipeline reconstructs tracks for quirk-like BSM particles using two sequential ML models:

1. **Metric Learning** (embedding network) — maps hits into a latent space where nearby points likely belong to the same track
2. **Graph Neural Network (GNN)** — classifies edges in the constructed graph as signal or background

Pre-trained weights for the `Lambda500_pre_selection_quirk` run are included under `artifacts/`.

---

## Repository Structure

```
Examples/QuirkTracking/Scripts/   # Main pipeline scripts (Steps 0–6)
  Step_0_dataset.py               # Preprocessing raw hits into feature store
  Step_1_Train_Metric_Learning.py # Train the embedding model
  Step_2_Run_Metric_Learning.py   # Run embedding inference to build graphs
  Step_3_Train_GNN.py             # Train the GNN
  Step_4_Run_GNN.py               # Run GNN inference to score edges
  Step_5_Build_Track_Candidates.py# Connected-components track building
  Step_6_Evaluate_Reconstruction.py# Evaluation and efficiency metrics
  utils/                          # Shared utility functions

Pipelines/TrackML_Example/        # Core Lightning modules (embedding, GNN, processing)
Architectures/                    # Standalone model architecture definitions
artifacts/Lambda500_pre_selection_quirk/
  metric_learning/quirk.ckpt      # Pre-trained embedding model (~49 MB)
  gnn/quirk.ckpt                  # Pre-trained GNN (~1.5 MB)
```

---

## Installation

Requires Python 3.9 and CUDA 11.3 (GPU) or CPU-only.

**GPU:**
```bash
conda env create -f gpu_environment.yml python=3.9
conda activate exatrkx-gpu
pip install -e .
```

**CPU only:**
```bash
conda env create -f cpu_environment.yml python=3.9
conda activate exatrkx-cpu
pip install -e .
```

---

## Configuration

Each step reads a single YAML config file. Create one with the following structure (paths must be absolute):

```yaml
common_configs:
  experiment_name: quirk
  artifact_directory: /path/to/your/artifacts/run_name
  gpus: 1
  clear_directories: False          # set True to wipe intermediate dirs between runs

metric_learning_configs:
  input_dir: /path/to/feature_store/run_name
  output_dir: /path/to/metric_learning_processed/run_name
  pt_signal_cut: 0.
  pt_background_cut: 0.
  train_split: [65000, 100, 3900]
  true_edges: modulewise_true_edges
  spatial_channels: 3
  cell_channels: 0
  emb_hidden: 1024
  nb_layer: 4
  emb_dim: 12
  activation: Tanh
  weight: 2
  randomisation: 2
  points_per_batch: 100000
  r_train: 0.1
  r_val: 0.1
  r_test: 0.1
  knn: 50
  warmup: 8
  margin: 0.1
  lr: 0.001
  factor: 0.7
  patience: 4
  regime: [rp, hnm, norm]
  max_epochs: 5

gnn_configs:
  input_dir: /path/to/metric_learning_processed/run_name
  output_dir: /path/to/gnn_processed/run_name
  edge_cut: 0.5
  pt_signal_min: 0.
  pt_background_min: 0.
  datatype_names: [train, val, test]
  datatype_split: [65000, 100, 3900]
  noise: False
  spatial_channels: 3
  cell_channels: 0
  hidden: 64
  n_graph_iters: 8
  nb_node_layer: 3
  nb_edge_layer: 3
  layernorm: True
  aggregation: sum_max
  hidden_activation: SiLU
  weight: 1
  warmup: 20
  lr: 0.002
  factor: 0.7
  patience: 100
  truth_key: pid_signal
  regime: []
  mask_background: True
  max_epochs: 5

track_building_configs:
  score_cut: 0.9
  output_dir: /path/to/trackbuilding_processed/run_name

evaluation_configs:
  output_dir: /path/to/evaluation/run_name
  min_pt: 0
  max_eta: 4
  min_track_length: 15
  min_particle_length: 22
  matching_fraction: 0.5
  matching_style: two_way
```

---

## Running the Pipeline

> **Note:** Before running, update the hardcoded `project_root` variable at the top of each script to point to your local checkout of this repository.

Each step is run from `Examples/QuirkTracking/Scripts/`, passing the config file as an argument:

```bash
cd Examples/QuirkTracking/Scripts/

# Step 0: Preprocess raw event CSVs into a feature store
python Step_0_dataset.py your_config.yaml

# Step 1: Train metric learning (skip if using pre-trained weights)
python Step_1_Train_Metric_Learning.py your_config.yaml

# Step 2: Run embedding inference to build graphs
python Step_2_Run_Metric_Learning.py your_config.yaml

# Step 3: Train GNN (skip if using pre-trained weights)
python Step_3_Train_GNN.py your_config.yaml

# Step 4: Run GNN inference to score graph edges
python Step_4_Run_GNN.py your_config.yaml

# Step 5: Build track candidates from scored graphs
python Step_5_Build_Track_Candidates.py your_config.yaml

# Step 6: Evaluate reconstruction performance
python Step_6_Evaluate_Reconstruction.py your_config.yaml
```

## Using Pre-trained Weights

To use the included pre-trained weights, point `artifact_directory` in your config to this repo's `artifacts/` folder and set the run name to `Lambda500_pre_selection_quirk`:

```yaml
common_configs:
  experiment_name: quirk
  artifact_directory: /path/to/this/repo/artifacts/Lambda500_pre_selection_quirk
```

Then skip Steps 1 and 3 (training) and run Steps 2, 4, 5, and 6 for inference only.

---

## Pipeline Diagram

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/application_diagram_1.png" width="600"/>
</figure>
</div>

---

## Credits

Built on the [ExaTrkX Tracking-ML](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX) framework by the Exa.TrkX Collaboration.
