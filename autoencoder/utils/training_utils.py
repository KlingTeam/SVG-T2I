"""Training utilities for setup and configuration"""

import os
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from ldm.util import instantiate_from_config
from utils.callbacks_allinone import SetupCallback, ImageLogger, CUDACallback, ImageLogger_native


def setup_trainer(config, logdir, resume_checkpoint=None):
    """Setup PyTorch Lightning trainer with callbacks"""
    
    # Trainer configuration
    trainer_config = config.training.trainer
    trainer_kwargs = {
        k: v for k, v in trainer_config.items() 
        if k not in ['callbacks', 'logger']
    }
    
    # Setup logger
    logger = setup_logger(config, logdir)
    trainer_kwargs["logger"] = logger
    
    # Setup callbacks
    callbacks = setup_callbacks(config, logdir)
    trainer_kwargs["callbacks"] = callbacks
    
    # Resume from checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        trainer_kwargs["resume_from_checkpoint"] = resume_checkpoint
    
    trainer = Trainer(**trainer_kwargs)
    trainer.logdir = logdir
    
    return trainer


def setup_logger(config, logdir):
    """Setup logger based on configuration"""
    
    logger_config = config.training.get("logger", {})
    if not logger_config:
        # Default tensorboard logger
        logger_config = {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": "tensorboard",
                "save_dir": logdir,
            }
        }
    
    return instantiate_from_config(logger_config)

def setup_callbacks(config, logdir):
    """Setup training callbacks"""
    
    callbacks = []
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    save_every_n_train_steps = config.training.get("save_every_n_train_steps", 20000)
    save_every_n_epochs = config.training.get("save_every_n_epochs", 1)
    image_logger_type = config.training.get("image_logger_type", "native")
    
    # 按步数保存的检查点回调
    step_checkpoint_config = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": os.path.join(ckptdir, "steps"),
            "filename": "{actual_step:06.0f}",
            "verbose": True,
            "save_last": False,  # 只在epoch检查点中保存last
            "every_n_train_steps": save_every_n_train_steps,
            "save_top_k": -1,    # 保存所有步数检查点
            "save_weights_only": True
        }
    }
    callbacks.append(instantiate_from_config(step_checkpoint_config))
    
    # 按epoch保存的检查点回调
    epoch_checkpoint_config = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": os.path.join(ckptdir, "epochs"),
            "filename": "{epoch:03}",
            "verbose": True,
            "save_last": True,   # 在这里保存last
            "every_n_epochs": save_every_n_epochs, # 每个epoch结束后保存
            "save_top_k": -1,    # 保存所有epoch检查点
            "save_weights_only": True
        }
    }
    callbacks.append(instantiate_from_config(epoch_checkpoint_config))
    
    # Learning rate monitor
    lr_monitor_config = {
        "target": "pytorch_lightning.callbacks.LearningRateMonitor",
        "params": {
            "logging_interval": "step"
        }
    }
    callbacks.append(instantiate_from_config(lr_monitor_config))
    
    # Image logger
    print(image_logger_type)
    if image_logger_type == "native":
        image_logger_config = {
            "target": "utils.callbacks_allinone.ImageLogger_native",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True,
                "rescale": True
            }
        }
    else:
        image_logger_config = {
            "target": "utils.callbacks_allinone.ImageLogger",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True,
                "rescale": True
            }
        }
    
    callbacks.append(instantiate_from_config(image_logger_config))
    
    # Setup callback
    setup_callback_config = {
        "target": "utils.callbacks_allinone.SetupCallback",
        "params": {
            "logdir": logdir,
            "ckptdir": ckptdir,
            "cfgdir": cfgdir
        }
    }
    callbacks.append(instantiate_from_config(setup_callback_config))
    
    # CUDA callback
    cuda_callback_config = {
        "target": "utils.callbacks_allinone.CUDACallback"
    }
    callbacks.append(instantiate_from_config(cuda_callback_config))
    
    return callbacks