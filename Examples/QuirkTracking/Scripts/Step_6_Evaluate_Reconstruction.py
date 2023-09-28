"""
This script runs step 6 of the TrackML Quickstart example: Evaluating the track reconstruction performance.
"""

import sys
import os
import yaml
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from functools import partial
from utils.convenience_utils import headline
from utils.plotting_utils import plot_pt_eff

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("5_Build_Track_Candidates.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def conformal_mapping(file):

    graph = torch.load(file, map_location="cpu")
    r_reco = graph.x[:,0].detach().numpy()
    phi_reco = graph.x[:,1].detach().numpy()
    z = graph.x[:,2].detach().numpy()
    x = r_reco * np.cos(phi_reco)
    y = r_reco * np.sin(phi_reco)
    
    hid = graph.hid
    tid = graph.labels
    pid = graph.pid
    true_pt = graph.pt
    event_file = graph.event_file[-4:]

    """
    x, y, z: np.array([])
    return: 
    """
    unique_particle_ids = np.unique(pid)  # get different 1particle_id
    
    for particle_id in unique_particle_ids:
        # ref. 10.1016/0168-9002(88)90722-X
        # choose the dataset with different particle_id
        mask = (pid == particle_id)
        x_particle = x[mask]
        #y_particle = y[mask] 
        y_particle = y[mask] - 0.001
        z_particle = z[mask]
        
        r = x_particle**2 + y_particle**2
        true_pt_particle = true_pt[mask]

        u = x_particle / r
        v = y_particle / r
        # assuming the imapact parameter is small
        # the v = 1/(2b) - u x a/b - u^2 x epsilon x (R/b)^3
        if len(u) < 6 or len(v) < 6: # Choose at least 6 hits.
            continue
        #print(f"u:{u}")
        #print(f"v:{v}")
        pp, vv = np.polyfit(u, v, 2, cov=True)
        b = 0.5/pp[2]
        a = -pp[1]*b
        R = np.sqrt(a**2 + b**2)
        e = -pp[0] / (R/b)**3 # approximately equals to d0
        dev = 2*e*R / b**2

        magnetic_field = 2.0
        pT = 0.3*magnetic_field*R # in MeV
        # print(a, b, R, e, pT)

        p_rz = np.polyfit(np.sqrt(r), z_particle, 2)
        y_p_rz = np.polyval(p_rz, np.sqrt(r))
        residuals = z_particle - y_p_rz
        chi_square = np.sum((residuals / y_p_rz)**2)
        
        y_p_rz = np.polyval(p_rz, np.sqrt(r))
    
        pp_rz = np.poly1d(p_rz)
        z0 = pp_rz(abs(e))

        r3 = np.sqrt(r + z_particle**2)
        p_zr = np.polyfit(r3, z_particle, 2)
    
        cos_val = p_zr[0]*z0 + p_zr[1]
        theta = np.arccos(cos_val)
        eta = -np.log(np.tan(theta/2.))
        phi = np.arctan2(b, a)
    
    parameter_track = pd.DataFrame({"hit_id": hid, "track_id": tid, "particle_id": pid, "x": x, "y": y, "z": z, "r": r_reco, "phi_reco": phi_reco, "e": e, "z0": z0, "eta": eta, "phi": phi,  "dev": dev, "cos_val": cos_val, "theta": theta, "mapping_pT": pT,  "event_file": event_file, "chi_square_rz": chi_square})
    
    return parameter_track


def load_reconstruction_df(file):
    """Load the reconstructed tracks from a file."""
    graph = torch.load(file, map_location="cpu")
    reconstruction_df = pd.DataFrame({"hit_id": graph.hid, "track_id": graph.labels, "particle_id": graph.pid})
    #print(reconstruction_df)
    return reconstruction_df

def load_particles_df(file):
    """Load the particles from a file."""
    graph = torch.load(file, map_location="cpu")

    # Get the particle dataframe
    particles_df = pd.DataFrame({"particle_id": graph.pid, "pt": graph.pt})

    # Reduce to only unique particle_ids
    particles_df = particles_df.drop_duplicates(subset=['particle_id'])
    #print(particles_df)
    return particles_df

def get_matching_df(reconstruction_df, particles_df, min_track_length=1, min_particle_length=1):
    
    # Get track lengths
    candidate_lengths = reconstruction_df.track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_reco_hits"})

    # Get true track lengths
    particle_lengths = reconstruction_df.drop_duplicates(subset=['hit_id']).particle_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"particle_id", "particle_id": "n_true_hits"})

    spacepoint_matching = reconstruction_df.groupby(['track_id', 'particle_id']).size()\
        .reset_index().rename(columns={0:"n_shared"})

    spacepoint_matching = spacepoint_matching.merge(candidate_lengths, on=['track_id'], how='left')
    spacepoint_matching = spacepoint_matching.merge(particle_lengths, on=['particle_id'], how='left')
    spacepoint_matching = spacepoint_matching.merge(particles_df, on=['particle_id'], how='left')

    # Filter out tracks with too few shared spacepoints
    spacepoint_matching["is_matchable"] = spacepoint_matching.n_reco_hits >= min_track_length
    spacepoint_matching["is_reconstructable"] = spacepoint_matching.n_true_hits >= min_particle_length
    spacepoint_matching["is_catchable"] = spacepoint_matching.n_true_hits - spacepoint_matching.n_reco_hits  <= 5
    
    return spacepoint_matching

def calculate_matching_fraction(spacepoint_matching_df):
    spacepoint_matching_df = spacepoint_matching_df.assign(
        purity_reco=np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_reco_hits))
    spacepoint_matching_df = spacepoint_matching_df.assign(
        eff_true = np.true_divide(spacepoint_matching_df.n_shared, spacepoint_matching_df.n_true_hits))

    return spacepoint_matching_df

def evaluate_labelled_graph(graph_file, matching_fraction=0.5, matching_style="ATLAS", min_track_length=1, min_particle_length=1):

    if matching_fraction < 0.5:
        raise ValueError("Matching fraction must be >= 0.5")

    if matching_fraction == 0.5:
        # Add a tiny bit of noise to the matching fraction to avoid double-matched tracks
        matching_fraction += 0.00001

    # Load the labelled graphs as reconstructed dataframes
    reconstruction_df = load_reconstruction_df(graph_file)
    particles_df = load_particles_df(graph_file)

    #print("--------------------------reconstruction_df----------------------------")
    #print(reconstruction_df)
    #print("--------------------------particle_df----------------------------")
    #print(particles_df)
    # Get matching dataframe 
    matching_df = get_matching_df(reconstruction_df, particles_df, min_track_length=min_track_length, min_particle_length=min_particle_length) 
    matching_df["event_id"] = int(graph_file.split("/")[-1])

    # calculate matching fraction
    matching_df = calculate_matching_fraction(matching_df)
     
    # Run matching depending on the matching style
    if matching_style == "ATLAS":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = matching_df.purity_reco >= matching_fraction
    elif matching_style == "one_way":
        matching_df["is_matched"] = matching_df.purity_reco >= matching_fraction
        matching_df["is_reconstructed"] = matching_df.eff_true >= matching_fraction
    elif matching_style == "two_way":
        matching_df["is_matched"] = matching_df["is_reconstructed"] = (matching_df.purity_reco >= matching_fraction) & (matching_df.eff_true >= matching_fraction)
    
    #if matching_df["is_matchable"].any():
    #    print("--------------------------matching_df----------------------------")
    #    print(matching_df)
    #    print("--------------------------reconstruction_df----------------------------")
    #    print(reconstruction_df)
    #    print("--------------------------particle_df----------------------------")
    #    print(particles_df)
    return matching_df

def evaluate(config_file="pipeline_config.yaml"):

    logging.info(headline("Step 6: Evaluating the track reconstruction performance"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)

    common_configs = all_configs["common_configs"]
    track_building_configs = all_configs["track_building_configs"]
    evaluation_configs = all_configs["evaluation_configs"]

    logging.info(headline("a) Loading labelled graphs"))

    input_dir = track_building_configs["output_dir"]
    output_dir = evaluation_configs["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_graph_files = os.listdir(input_dir)
    all_graph_files = [os.path.join(input_dir, graph) for graph in all_graph_files]

    evaluated_events = []
    events_parameters_track = []
    reconstruction_hit = []
    for graph_file in tqdm(all_graph_files):
        evaluated_events.append(evaluate_labelled_graph(graph_file, 
                                matching_fraction=evaluation_configs["matching_fraction"], 
                                matching_style=evaluation_configs["matching_style"],
                                min_track_length=evaluation_configs["min_track_length"],
                                min_particle_length=evaluation_configs["min_particle_length"]))
        events_parameters_track.append(conformal_mapping(graph_file))
    evaluated_events = pd.concat(evaluated_events)
    events_parameters_track = pd.concat(events_parameters_track)

    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    #reconstructed_particles = particles[particles["is_reconstructed"] & particles["is_matchable"] & particles["is_catchable"]]
    
    #With reconstructed
    reconstructed_particles = particles[particles["is_reconstructed"] & particles["is_matchable"] ]
    reconstruct_data = pd.merge(reconstructed_particles, events_parameters_track, on=["track_id", "particle_id"])
    columns_to_drop = ["is_reconstructable", "is_matchable","is_catchable","is_matched","is_reconstructed"]
    reconstruct_data = reconstruct_data.drop(columns=columns_to_drop)
    #print(reconstruct_data)
    reconstruct_data.to_csv("./output/track_prue_quirk_val_1000.csv", index=False)
   
    #Without reconstructed
    particles_data = pd.merge(particles, events_parameters_track, on=["track_id", "particle_id"])
    columns_to_drop = ["is_reconstructable", "is_matchable","is_catchable","is_matched","is_reconstructed"]
    particles_data = particles_data.drop(columns=columns_to_drop)
    particles_data.to_csv("./output/track_bkg_2000_no_reco.csv", index=False) 
     
    tracks = evaluated_events[evaluated_events["is_matchable"]]
    matched_tracks = tracks[tracks["is_matched"]]

    n_particles = len(particles.drop_duplicates(subset=['event_id', 'particle_id']))
    n_reconstructed_particles = len(reconstructed_particles.drop_duplicates(subset=['event_id', 'particle_id']))
    
    n_tracks = len(tracks.drop_duplicates(subset=['event_id', 'track_id']))
    n_matched_tracks = len(matched_tracks.drop_duplicates(subset=['event_id', 'track_id']))
    
    n_dup_reconstructed_particles = len(reconstructed_particles) - n_reconstructed_particles

    logging.info(headline("b) Calculating the performance metrics"))
    logging.info(f"Number of reconstructed particles: {n_reconstructed_particles}")
    logging.info(f"Number of particles: {n_particles}")
    logging.info(f"Number of matched tracks: {n_matched_tracks}")
    logging.info(f"Number of tracks: {n_tracks}")
    logging.info(f"Number of duplicate reconstructed particles: {n_dup_reconstructed_particles}")   

    # Plot the results across pT and eta
    eff = n_reconstructed_particles / n_particles
    fake_rate = 1 - (n_matched_tracks / n_tracks)
    dup_rate = n_dup_reconstructed_particles / n_reconstructed_particles
    
    logging.info(f"Efficiency: {eff:.3f}")
    logging.info(f"Fake rate: {fake_rate:.3f}")
    logging.info(f"Duplication rate: {dup_rate:.3f}")

    logging.info(headline("c) Plotting results"))

    # First get the list of particles without duplicates
    grouped_reco_particles = particles.groupby('particle_id')["is_reconstructed"].any()
    particles["is_reconstructed"] = particles["particle_id"].isin(grouped_reco_particles[grouped_reco_particles].index.values)
    particles = particles.drop_duplicates(subset=['particle_id'])

    # Plot the results across pT and eta
    plot_pt_eff(particles)

    # TODO: Plot the results
    return evaluated_events, reconstructed_particles, particles, matched_tracks, tracks, reconstruct_data, particles_data

if __name__ == "__main__":

    args = parse_args()
    config_file = args.config
    
    #load_particles_info(config_file)

    evaluate(config_file) 