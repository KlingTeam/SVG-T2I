"""Custom callbacks for training"""

import os
import time
import numpy as np
from PIL import Image
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf


class SetupCallback(Callback):
    """Callback to setup logging directories and save configs"""
    
    def __init__(self, logdir, ckptdir, cfgdir):
        super().__init__()
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir

    @rank_zero_only
    def on_pretrain_routine_start(self, trainer, pl_module):
        """Create directories and save configs at start of training"""
        
        # Create directories
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)
        
        print(f"Training setup complete. Logdir: {self.logdir}")


# ==============================
# ✅ Base Class: General Image Logging Logic
# ==============================
class BaseImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs or {}
        self.log_first_step = log_first_step

        # Incremental logging steps
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]

        # Logger → handler mapping
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._log_to_tensorboard_default,
        }

    # ---------- Public Interface ----------
    @rank_zero_only
    def _log_to_tensorboard_default(self, pl_module, images, batch_idx, split):
        """Default TensorBoard writer"""
        writer = getattr(pl_module.logger, "experiment", None)
        if writer is None:
            return
        for k, v in images.items():
            grid = torchvision.utils.make_grid(v)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            writer.add_image(f"{split}/{k}", grid, global_step=pl_module.global_step)

    @rank_zero_only
    def save_images(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """General save interface (fixed grid)"""
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        for k, img in images.items():
            grid = torchvision.utils.make_grid(img, nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            Image.fromarray(grid).save(os.path.join(root, filename))

    def _prepare_images(self, images):
        """Clamp + detach + take first N images"""
        processed = {}

        for k, img in images.items():
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()
                if self.clamp:
                    img = torch.clamp(img, -1., 1.)
                processed[k] = img[:self.max_images]
            else:
                imgs_processed = []
                for im in img:
                    im = torch.clamp(im, -1., 1.).detach().cpu()
                    imgs_processed.append(im)
                processed[k] = imgs_processed[:self.max_images]

        return processed

    # ---------- Main Logic ----------
    @rank_zero_only
    def log_images(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if not self._should_log(check_idx):
            return
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return

        was_training = pl_module.training
        if was_training:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

        images = self._prepare_images(images)

        # Saving + TensorBoard logging
        self.save_images(pl_module.logger.save_dir, split, images,
                        pl_module.global_step, pl_module.current_epoch, batch_idx)
        log_func = self.logger_log_images.get(type(pl_module.logger), None)
        if log_func:
            log_func(pl_module, images, batch_idx, split)

        if was_training:
            pl_module.train()

    def _should_log(self, idx):
        """Determine whether it's time to log"""
        if ((idx % self.batch_freq) == 0 or (idx in self.log_steps)) and (idx > 0 or self.log_first_step):
            if self.log_steps:
                self.log_steps.pop(0)
            return True
        return False

    # ---------- Lightning Hooks ----------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_images(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and pl_module.global_step > 0:
            self.log_images(pl_module, batch, batch_idx, split="val")


# ==============================
# ✅ Simplified Version: Basic Logging Only
# ==============================
class ImageLogger(BaseImageLogger):
    """Minimal Image Logger (fixed grid + TensorBoard)"""
    pass


# ==============================
# ✅ Advanced Version: Variable Resolution & Input/Reconstruction Pairs
# ==============================
class ImageLogger_native(BaseImageLogger):
    """Enhanced ImageLogger: supports variable resolutions and input/output comparison grids"""

    @rank_zero_only
    def save_images(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """Save images with variable sizes"""
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        for k, img in images.items():
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    for i in range(img.shape[0]):
                        self._save_single_image(img[i], root, k, i, global_step, current_epoch, batch_idx)
                else:
                    self._save_single_image(img, root, k, 0, global_step, current_epoch, batch_idx)
            else:
                for i in range(len(img)):
                    self._save_single_image(img[i][0], root, k, i, global_step, current_epoch, batch_idx)

    def _save_single_image(self, img, root, key, idx, global_step, current_epoch, batch_idx):
        if self.rescale:
            img = (img + 1.0) / 2.0
        img = torch.clamp(img, 0, 1)
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]

        np_img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        filename = f"{key}_idx-{idx}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
        Image.fromarray(np_img).save(os.path.join(root, filename))

    @rank_zero_only
    def _log_to_tensorboard_default(self, pl_module, images, batch_idx, split):
        """Write side-by-side input/reconstruction pairs to TensorBoard"""
        writer = getattr(pl_module.logger, "experiment", None)
        if writer is None:
            return

        keys = list(images.keys())
        input_key = next((k for k in keys if "input" in k.lower()), None)
        recon_key = next((k for k in keys if any(x in k.lower() for x in ["recon", "dec", "output"])), None)

        if not input_key or not recon_key:
            print(f"[ImageLogger_native] Warning: cannot find input/reconstruction keys in {keys}")
            return

        inputs, recons = images[input_key], images[recon_key]
        if isinstance(inputs, torch.Tensor):
            length = min(inputs.shape[0], recons.shape[0])
        else:
            length = min(len(inputs), len(recons))

        n = min(length, self.max_images)

        for i in range(n):
            inp = inputs[i].detach().cpu()
            rec = recons[i].detach().cpu()

            if inp.dim() == 4:
                inp = inp[0]
            if rec.dim() == 4:
                rec = rec[0]

            if self.rescale:
                inp = (inp + 1.0) / 2.0
                rec = (rec + 1.0) / 2.0
            inp = torch.clamp(inp, 0, 1)
            rec = torch.clamp(rec, 0, 1)

            if inp.shape[1:] != rec.shape[1:]:
                h, w = min(inp.shape[1], rec.shape[1]), min(inp.shape[2], rec.shape[2])
                inp, rec = inp[:, :h, :w], rec[:, :h, :w]

            grid = torchvision.utils.make_grid([inp, rec], nrow=2, padding=2)
            writer.add_image(f"{split}/pair_{i}", grid, global_step=pl_module.global_step)
        writer.flush()


class CUDACallback(Callback):
    """Callback to monitor GPU memory and timing"""
    
    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time
        
        print(f"Epoch time: {epoch_time:.2f}s, Peak memory: {max_memory:.2f}MB")

