"""
This script runs step 1 of the TrackML Quickstart example: Training the metric learning model.
"""

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import signal

CONFIG = os.getenv('CONFIG', 'default_value4')

# Set the project root directory
project_root = "/global/homes/l/lcondren/pipeline_copy"
sys.path.append(project_root)

CONFIG = os.path.join(project_root, CONFIG)
# Remove the incorrect path if it exists
incorrect_path = "/global/u2/l/lcondren/Tracking_pipeline"
if incorrect_path in sys.path:
    sys.path.remove(incorrect_path)


sys.path.append("../../")
print("system path",sys.path)

# os.environ["CUDA_HOME"] = "/pscratch/sd/l/lcondren/exatrkx-gpu2"
# os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
# os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

print(f"CUDA Version: {torch.version.cuda}")
print(torch.__version__)
from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
from utils.convenience_utils import headline

print(f"LayerlessEmbedding is imported from: {os.path.abspath(LayerlessEmbedding.__module__.replace('.', '/') + '.py')}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("1_Train_Metric_Learning.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default=CONFIG)
    return parser.parse_args()


def train(config_file=CONFIG):

    logging.info(headline("Step 1: Running metric learning training"))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    metric_learning_configs = all_configs["metric_learning_configs"]

    logging.info(headline("a) Initialising model"))

    model = LayerlessEmbedding(metric_learning_configs)
    logging.info(headline("b) Running training" ))
    model.setup(stage='fit')
    #print(model.testset)

    save_directory = os.path.join(common_configs["artifact_directory"], "metric_learning")
    logger = CSVLogger(save_directory, name=common_configs["experiment_name"], version=0)
    checkpoint_path = os.path.join(save_directory, common_configs["experiment_name"]+".ckpt")
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else None,
        gpus=common_configs["gpus"],
        max_epochs=metric_learning_configs["max_epochs"],
        logger=logger,
        default_root_dir=save_directory,
        resume_from_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=save_directory,
                filename=common_configs["experiment_name"] + "-{epoch:02d}-{step:04d}",
                save_top_k=-1,  # Save all checkpoints (can be adjusted to save only recent ones)
                every_n_train_steps=200  # Adjust as needed for frequency
            )
        ]
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

    trainer, model = train(config_file)    

