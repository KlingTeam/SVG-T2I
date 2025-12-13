"""
Training Script for DINO-based Autoencoder

Refactored version with clean separation of configuration and training logic.
"""

import os
import argparse
import datetime
import glob
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from utils.training_utils import setup_trainer
from utils.data_utils import create_data_module, get_dataset_info, setup_data_module


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to main config file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup logging directory
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    # logdir = os.path.join(config.training.logdir, f"{now}_{cfg_name}")
    logdir = os.path.join(config.training.logdir, f"{cfg_name}")
    # print(cfg_name)
    print(logdir)

    # Set random seed
    seed_everything(args.seed)

    # print(f"Try to resume from {logdir}")
    # ckpt_files = glob.glob(os.path.join(logdir, "checkpoints", "epochs", "epoch=*.ckpt"))
    # if not ckpt_files:
    #     print(f"Warning: No checkpoint files found in {os.path.join(logdir, 'checkpoints')}, training from scratch")
    #     ckpt = None
    # else:
    #     ckpt_files.sort(key=lambda x: int(os.path.basename(x).split('=')[1].split('.')[0]))
    #     ckpt = ckpt_files[-1]
    # resume_from_checkpoint = ckpt

    # Create data module using original structure
    print("Creating data module...")
    data_module = create_data_module(config.data)
    
    # Setup data (this calls prepare_data() and setup())
    print("Setting up data...")
    data_module = setup_data_module(data_module)
    
    # Print dataset information
    get_dataset_info(data_module)
    
    # Create model
    print("Creating model...")
    model = instantiate_from_config(config.model)
    
    # Load initial weights if specified
    if config.model.get("init_weight"):
        init_weight_path = config.model.init_weight
        print(f"Loading initial weights from {init_weight_path}")
        try:
            if os.path.exists(init_weight_path):
                checkpoint = torch.load(init_weight_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                print("Initial weights loaded successfully")
            else:
                print(f"Warning: Initial weights file not found: {init_weight_path}")
        except Exception as e:
            print(f"Error loading initial weights: {e}")
    
    # Setup learning rate
    setup_learning_rate(model, config)

    # Setup trainer
    trainer = setup_trainer(config, logdir)

    # Start training
    print("Starting training...")
    try:
        print("Starting training run")
        # trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)
        trainer.fit(model, data_module)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


def setup_learning_rate(model, config):
    """Configure learning rate based on config"""
    base_lr = config.model.base_learning_rate
    accumulate_grad_batches = config.training.get("accumulate_grad_batches", 1)
    
    if config.training.get("scale_lr", False):
        batch_size = config.data.params.batch_size
        devices = config.training.trainer.devices
        model.learning_rate = accumulate_grad_batches * devices * batch_size * base_lr
        print(f"Scaled learning rate to {model.learning_rate:.2e}")
    else:
        model.learning_rate = base_lr
        print(f"Using base learning rate: {model.learning_rate:.2e}")


if __name__ == "__main__":
    main()