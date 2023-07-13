"""Utilities for processing the overall event.

The module contains useful functions for handling data at the event level. More fine-grained utilities are 
reserved for `detector_utils` and `cell_utils`.
    
Todo:
    * Pull module IDs out into a csv file for readability """

# System
import os
import logging

# Externals
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data
import itertools


def get_layerwise_edges(hits):

    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)
    hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan
    hit_list = (
        hits.groupby(["particle_id", "layer"], sort=False)["index"]
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    true_edges = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j)))
    true_edges = np.array(true_edges).T

    return true_edges, hits


def get_modulewise_edges(hits):

    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    signal = signal.drop_duplicates(
        subset=["particle_id", "layer_id"]
    )

    # Sort by increasing distance from production
    signal = signal.assign(
        R=np.sqrt(
            (signal.x - signal.vx) ** 2
            + (signal.y - signal.vy) ** 2
            + (signal.z - signal.vz) ** 2
        )
    )
    signal = signal.sort_values("R").reset_index(drop=False)

    # Handle re-indexing
    signal = signal.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particle ID
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )

    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])

    true_edges = np.array(true_edges).T

    true_edges = signal.unsorted_index.values[true_edges]

    return true_edges


def select_hits(truth, particles, endcaps=False, noise=False, min_pt=None):
    # Barrel volume and layer ids
    vlids = [
        (1),
        (2),
        (3),
        (4),
        (5),
        (6),
        (7),
        (8),
    ]
    n_det_layers = len(vlids)
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = truth.groupby(["layer_id"])
    truth = pd.concat(
        [vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)]
    )

    if noise:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "px", "py", "pz"]], on="particle_id", how="left"
        )
    else:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "px", "py", "pz"]], on="particle_id", how="inner"
        )

    truth = truth.assign(pt=np.sqrt(truth.px**2 + truth.py**2))

    if min_pt:
        truth = truth[truth.pt > min_pt]

    # Calculate derived hits variables
    x = truth.r*np.sin(truth.phi)
    y = truth.r*np.cos(truth.phi)
    z = truth.z
    #r = np.sqrt(hits.x**2 + hits.y**2)
    #phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    truth = truth.assign(x=x, y=y)

    return truth


def build_event(
    event_file,
    feature_scale,
    endcaps=False,
    modulewise=True,
    layerwise=True,
    noise=False,
    min_pt=None,
):
    hits_file = event_file + "_hits.csv"
    particles_file = event_file + "_particles.csv"
    truth, particles = pd.read_csv(hits_file), pd.read_csv(particles_file)

    truth = select_hits(truth, particles, endcaps=endcaps, noise=noise, min_pt=min_pt).assign(
        evtid=int(event_file[-9:])
    )

    # Handle which truth graph(s) are being produced
    modulewise_true_edges, layerwise_true_edges = None, None

    if layerwise:
        layerwise_true_edges, hits = get_layerwise_edges(truth)
        logging.info(
            "Layerwise truth graph built for {} with size {}".format(
                event_file, layerwise_true_edges.shape
            )
        )

    if modulewise:
        modulewise_true_edges = get_modulewise_edges(truth)
        logging.info(
            "Modulewise truth graph built for {} with size {}".format(
                event_file, modulewise_true_edges.shape
            )
        )

    logging.info("Weights constructed")

    return (
        truth[["r", "phi", "z"]].to_numpy() / feature_scale,
        truth.particle_id.to_numpy(),
        modulewise_true_edges,
        layerwise_true_edges,
        truth["hit_id"].to_numpy(),
        truth.pt.to_numpy(),
    )


def prepare_event(
    event_file,
    output_dir=None,
    endcaps=False,
    modulewise=True,
    layerwise=True,
    noise=False,
    min_pt=1,
    cell_information=False,
    overwrite=False,
    **kwargs
):
    try:
        evtid = int(event_file[-9:])
        filename = os.path.join(output_dir, str(evtid))
        print("evtid&filename:")
        print(evtid)
        print(filename)
        print("---")
        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing event {}".format(evtid))
            feature_scale = [1000, np.pi, 1000]

            (
                X,
                pid,
                modulewise_true_edges,
                layerwise_true_edges,
                hid,
                pt,
            ) = build_event(
                event_file,
                feature_scale,
                endcaps=endcaps,
                modulewise=modulewise,
                layerwise=layerwise,
                noise=noise,
                min_pt=min_pt,
            )
            print("event_file:")
            print(event_file)
            print("pid:")
            print(pid)
            data = Data(
                x=torch.from_numpy(X).float(),
                pid=torch.from_numpy(pid),
                event_file=event_file,
                hid=torch.from_numpy(hid),
                pt=torch.from_numpy(pt),
            )
            print(data)
            if modulewise_true_edges is not None:
                data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)
            if layerwise_true_edges is not None:
                data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)
            logging.info("Getting cell info")


            with open(filename, "wb") as pickle_file:
                torch.save(data, pickle_file)

        else:
            logging.info("{} already exists".format(evtid))
    except Exception as inst:
        print("File:", event_file, "had exception", inst)
