"""
This script runs step 3 of the TrackML Quickstart example: Training the graph neural network.
"""

import sys
import os
import yaml
import argparse
import logging
import signal
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

CONFIG = os.getenv('CONFIG', 'default_value4')


# Set the project root directory
project_root = "/global/homes/l/lcondren/pipeline_copy"
sys.path.append(project_root)

sys.path.append("../../")

from Pipelines.TrackML_Example.LightningModules.GNN.Models.interaction_gnn import InteractionGNN
from utils.convenience_utils import headline
print("system path",sys.path)
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("3_Train_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default=CONFIG)
    return parser.parse_args()


def train(config_file=CONFIG):

    logging.info(headline(" Step 3: Running GNN training "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    gnn_configs = all_configs["gnn_configs"]

    logging.info(headline("a) Initialising model" ))

    model = InteractionGNN(gnn_configs)

    logging.info(headline( "b) Running training" ))

    save_directory = os.path.join(common_configs["artifact_directory"], "gnn")
    logger = CSVLogger(save_directory, name=common_configs["experiment_name"], version=0)
    # Add early stopping callback

    early_stopping = EarlyStopping(
        monitor='train_loss',  # Metric to monitor
        patience=3,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Minimize the monitored metric
    )

    checkpoint_path = os.path.join(save_directory, common_configs["experiment_name"]+".ckpt")
    trainer = Trainer(
        gpus=common_configs["gpus"],
        max_epochs=gnn_configs["max_epochs"],
        logger=logger,
        default_root_dir=save_directory,

        resume_from_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None,
        enable_checkpointing=True,
        callbacks=[early_stopping, LearningRateMonitor(logging_interval='epoch')],  # Add the early stopping and learning rate monitor callbacks
        gradient_clip_val=0.5  
    )

    
    os.makedirs(save_directory, exist_ok=True)
    
    
    signal.signal(signal.SIGTERM, lambda signal_number, frame: (
        print("SIGTERM received. Saving checkpoint..."),
        trainer.save_checkpoint(checkpoint_path),
        sys.exit(0)  # Exit gracefully
    ))

    if os.path.exists(checkpoint_path):
        trainer.fit(model, ckpt_path=checkpoint_path)

    else:
        trainer.fit(model)
    logging.info(headline("c) Saving model") )

    print(f"checkpoint will be saved to: {checkpoint_path}")
    trainer.save_checkpoint(checkpoint_path)

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config

    train(config_file)    

