"""
This script runs step 5 of the TrackML Quickstart example: Labelling spacepoints based on the scored graph.
"""
import pandas as pd
import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import torch
import numpy as np
import scipy.sparse as sps

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from functools import partial

CONFIG = os.getenv('CONFIG', 'default_value4')


# Set the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from utils.convenience_utils import headline, delete_directory

sys.path.append("../../")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("5_Build_Track_Candidates.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default=CONFIG)
    return parser.parse_args()


def label_graph(graph, score_cut=0.8, save_dir="datasets/quickstart_track_building_processed"):

    #check if the graph that is read in has signal. check completed on lines 62 - 63
    event_num = os.path.basename(graph.event_file) + '-particles.csv'
    event_df = pd.read_csv(os.path.join("/pscratch/sd/l/lcondren/combined_hit_particle_files/train_19_test_10",event_num))
    sig_df = event_df[event_df['PID']==15]
    sig_id = sig_df['particle_id'].iloc[0]


    os.makedirs(save_dir, exist_ok=True)
    #print("max event score", max(graph.scores), "min event score",min(graph.scores))
    edge_mask = graph.scores > score_cut

    row, col = graph.edge_index[:, edge_mask]
    edge_attr = np.ones(row.size(0))

    N = graph.x.size(0)
    sparse_edges = sps.coo_matrix((edge_attr, (row.numpy(), col.numpy())), (N, N))
    #print("sparse edges",sparse_edges)
    _, candidate_labels = sps.csgraph.connected_components(sparse_edges, directed=False, return_labels=True)
    #print("candidate labels",candidate_labels)  
    graph.labels = torch.from_numpy(candidate_labels).long()
    
    particles_df = pd.DataFrame({"particle_id": graph.pid, "pt": graph.pt, "hit_id": graph.hid, "track_id": graph.labels})
    sig_df = particles_df[particles_df['particle_id']==sig_id]
    #print("sig df",sig_df)
    if not sig_df.empty:
        print(f"event {int(graph.event_file[-6:]) - 10000000} has signal")
    else:
        print(f"event {int(graph.event_file[-6:]) - 10000000} DOES NOT HAVE SIGNAL")
    
    torch.save(graph, os.path.join(save_dir, graph.event_file[-6:]))


def train(config_file=CONFIG):

    logging.info(headline( " Step 5: Building track candidates from the scored graph " ))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]
    track_building_configs = all_configs["track_building_configs"]

    logging.info(headline("a) Loading scored graphs" ))

    all_graphs = []
    for subdir in ["train", "val", "test"]:
    #for subdir in ["train"]:
   # for subdir in ["val", "test"]:
        subdir_graphs = os.listdir(os.path.join(gnn_configs["output_dir"], subdir))
        all_graphs += [torch.load(os.path.join(gnn_configs["output_dir"], subdir, graph), map_location="cpu") for graph in subdir_graphs]

    logging.info(headline( "b) Labelling graph nodes" ) )

    score_cut = track_building_configs["score_cut"]
    save_dir = track_building_configs["output_dir"]
    
    if common_configs["clear_directories"]:
        delete_directory(track_building_configs["output_dir"])

    # RUN IN SERIAL FOR NOW -->
    for graph in tqdm(all_graphs):
        label_graph(graph, score_cut=score_cut, save_dir=save_dir)



if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file) 
