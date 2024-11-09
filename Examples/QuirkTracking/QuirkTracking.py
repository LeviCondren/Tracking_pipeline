import sys, os
sys.path.append("../../")
from Scripts import train_metric_learning, run_metric_learning_inference, train_gnn, run_gnn_inference, build_track_candidates, evaluate_candidates
from Scripts.utils.convenience_utils import get_example_data, plot_true_graph, plot_true_graph_select,get_training_metrics, plot_training_metrics, plot_neighbor_performance, plot_predicted_graph, plot_track_lengths, plot_edge_performance, plot_graph_sizes
import yaml

import warnings
warnings.filterwarnings("ignore")
CONFIG = 'Scripts/pipeline_config_mix.yaml'

import torch
import numpy

data = torch.load("Scripts/Tracks_output/1")

with open(CONFIG, 'r') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

example_data_df, example_data_pyg = get_example_data(configs)
example_data_df.head()