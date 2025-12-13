# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict, defaultdict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time
import warnings
import torch.nn.functional as F

from PIL import Image
# import cairosvg
from diffusers import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    BackwardPrefetch
)


from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from data import DataNoReportException, ItemProcessor, MyDataset, read_general
from util import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
import models
from util import distributed_init, get_intra_node_process_group
from transport import create_transport
from util import instantiate_from_config, SmoothedValue
from omegaconf import OmegaConf

#############################################################################
#                            Data item Processor                            #
#############################################################################

class NonRGBError(DataNoReportException):
    pass

class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform, use_long_cap=False):
        self.image_transform = transform
        self.special_format_set = set()
        if use_long_cap:
            self.text_len_weights = [0.0, 0.0, 1.0]
        else:
            self.text_len_weights = [0.1, 0.35, 0.55]


    def process_item(self, data_item, training_mode=False):
        if "super_high_quality_caption" in data_item:
            url = data_item["image_path"]
            image = Image.open(read_general(url))
            text = data_item["super_high_quality_caption"]
            system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> "  # noqa
        elif "path" in data_item:
            url = data_item["path"]
            image = Image.open(read_general(url))
            text = data_item["prompt"]
            system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> "  
        else:
            raise ValueError(f"Unrecognized item: {data_item}")
            
        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            if mode not in self.special_format_set:
                self.special_format_set.add(mode)
                print(mode, url)
            if mode == "RGBA":
                image = to_rgb_if_rgba(image)
            elif mode == "P" or mode == "L":
                image = image.convert("RGB")
            else:
                raise NonRGBError()
            
        image = self.image_transform(image)

        if text is None or text.strip() == "":
            text = ""
        text = system_prompt + text

        return image, text


#############################################################################
#                           Training Helper Functions                       #
#############################################################################

def apply_average_pool(latent, factor):
    """
    Apply average pooling to downsample the latent.

    Args:
        latent (torch.Tensor): Latent tensor with shape (1, C, H, W).
        factor (int): Downsampling factor.

    Returns:
        torch.Tensor: Downsampled latent tensor.
    """
    return F.avg_pool2d(latent, kernel_size=factor, stride=factor)

def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    caps = [x[1] for x in samples]
    return image, caps

def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.95):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 显式开启 overlap

    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True, max_cap_len=256):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=max_cap_len,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

#############################################################################
#                           Timing Measurement Class                        #
#############################################################################

class TimeMeasurer:
    def __init__(self, tb_logger=None):
        self.durations = {}  # 直接存储持续时间
        self.tb_logger = tb_logger
        self.current_step = 0
        
    def start(self, name):
        # 在方案2中，start方法可以留空，因为我们在scope内部处理计时
        pass
        
    def end(self, name, step=None, duration=None):
        """记录持续时间"""
        if duration is not None:
            self.durations[name] = duration
            
        if step is not None:
            self.current_step = step
            
        # 记录到TensorBoard
        if self.tb_logger is not None and name in self.durations:
            self.tb_logger.add_scalar(f"timing/{name}", self.durations[name], self.current_step)
            
        return self.durations.get(name, 0.0)
        
    def measure_scope(self, name, step=None):
        return TimerScope(self, name, step)
    
    def get_duration(self, name):
        """获取特定阶段的持续时间"""
        return self.durations.get(name, 0.0)
    
    def print_time_distribution(self, logger, step, global_bsz, total_time):
        """打印时间分布"""
        if dist.get_rank() == 0:  # 只在主进程打印
            logger.info(f"Step {step} Time Distribution:")
            
            # 使用 get_duration 方法获取持续时间
            data_prefetch = self.get_duration("data_prefetch")
            dino_preprocessing = self.get_duration("dino_preprocessing")
            text_encoding = self.get_duration("text_encoding")
            model_forward = self.get_duration("model_forward")
            model_backward = self.get_duration("model_backward")
            grad_clip = self.get_duration("grad_clip")
            optimizer_step = self.get_duration("optimizer_step")
            
            measured_time = (data_prefetch + dino_preprocessing + text_encoding + 
                        model_forward + model_backward + grad_clip + optimizer_step)
            other_time = max(0, total_time - measured_time)

            self.tb_logger.add_scalar(f"timing/measured_time", measured_time, step)
            self.tb_logger.add_scalar(f"timing/other_time", other_time, step)

            logger.info(f"  Data Prefetch: {data_prefetch:.3f}s")
            logger.info(f"  DINO Preprocessing: {dino_preprocessing:.3f}s")
            logger.info(f"  Text Encoding: {text_encoding:.3f}s")
            logger.info(f"  Model Forward: {model_forward:.3f}s")
            logger.info(f"  Model Backward: {model_backward:.3f}s")
            logger.info(f"  Grad Clip: {grad_clip:.3f}s")
            logger.info(f"  Optimizer Step: {optimizer_step:.3f}s")
            logger.info(f"  Other: {other_time:.3f}s")
            logger.info(f"  Total: {total_time:.3f}s")
            logger.info(f"  Imgs/Sec: {global_bsz / total_time:.1f}")


class TimerScope:
    def __init__(self, measurer, name, step=None):
        self.measurer = measurer
        self.name = name
        self.step = step
        self.start_time = None
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        elapsed = time() - self.start_time
        self.measurer.end(self.name, self.step, duration=elapsed)


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    print(f"Starting rank={rank}, seed={seed}, "
          f"world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(
            os.path.join(
                args.results_dir, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname()
            )
        )
        # Initialize time measurer
        time_measurer = TimeMeasurer(tb_logger)
    else:
        logger = create_logger(None)
        tb_logger = None
        time_measurer = TimeMeasurer()

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    logger.info(f"Setting-up language model: google/gemma-2-2b")

    # create tokenizers
    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.gemma2b, token=args.hf_token)
    tokenizer.padding_side = "right"

    # create text encoders
    # text_encoder
    text_encoder = AutoModel.from_pretrained(
        args.gemma2b,
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
    ).cuda()
    text_encoder = setup_lm_fsdp_sync(text_encoder)
    logger.info(f"text encoder: {type(text_encoder)}")
    cap_feat_dim = text_encoder.config.hidden_size

    # Create model:
    model = models.__dict__[args.model](
        in_channels=384,
        qk_norm=args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{0:02d}-of-{1:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.get_checkpointing_wrap_module_list())
        checkpointing_list_ema = list(model_ema.get_checkpointing_wrap_module_list())
    else:
        checkpointing_list = []
        checkpointing_list_ema = []

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    # checkpointing (part2, after FSDP wrapping)
    if args.checkpointing:
        logger.info("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )

    logger.info(f"model:\n{model}\n")


    encoder_config = OmegaConf.load(args.autoencoder_path)
    encoder_config.model.params.is_train = False
    dinov3 = instantiate_from_config(encoder_config.model).cuda().eval()
    z_channels = encoder_config.model.params.ddconfig.z_channels
    print(z_channels)

    num_layer = encoder_config.model.params.get("num_layer", -1)
    dinov3_stats = torch.load("pre-trained/dinov3_s16p_layer_patchtoken_stats_dist.pt")
    dinov3_sp_mean = dinov3_stats["mean"][num_layer][None, None, :].to(device)
    dinov3_sp_std = dinov3_stats["std"][num_layer][None, None, :].to(device)
    if encoder_config.model.params.ddconfig.z_channels == 392:
        mean_extra = dinov3_sp_mean.mean(-1, keepdim=True).expand(-1, -1, 8).to(device)
        std_extra = dinov3_sp_std.mean(-1, keepdim=True).expand(-1, -1, 8).to(device)
        dinov3_sp_mean = torch.cat([dinov3_sp_mean, mean_extra], dim=-1).to(device)
        dinov3_sp_std = torch.cat([dinov3_sp_std, std_extra], dim=-1).to(device)

    logger.info("AdamW eps 1e-15 betas (0.9, 0.95)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-15, betas=(0.9, 0.95))
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    data_collection = {}
    for train_res in [args.resol]:
        logger.info(f"Creating data for resolution {train_res}")

        global_bsz = getattr(args, f"global_bsz")
        local_bsz = global_bsz // dp_world_size  # todo caution for sequence parallel
        micro_bsz = getattr(args, f"micro_bsz")
        assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
        logger.info(f"Global bsz: {global_bsz} Local bsz: {local_bsz} Micro bsz: {micro_bsz}")

        downsample_ratio = 16
        patch_size = downsample_ratio * model_patch_size
        logger.info(f"patch size: {patch_size}")
        max_num_patches = round((train_res / patch_size) ** 2)
        logger.info(f"Limiting number of patches to {max_num_patches}.")

        max_ratio = 1.0 if args.single_shape else 4.0
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size, max_ratio)
        logger.info("List of crop sizes:")
        for i in range(0, len(crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i : i + 6]]))
        image_transform = transforms.Compose(
            [
                transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True),
            ]
        )
        dataset = MyDataset(
            args.data_path,
            item_processor=T2IItemProcessor(image_transform),
        )
        num_samples = global_bsz * args.max_steps
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        logger.info(f"Total # samples to consume: {num_samples:,} " f"({num_samples / len(dataset):.2f} epochs)")
        sampler = get_train_sampler(
            dataset,
            dp_rank,
            dp_world_size,
            global_bsz,
            args.max_steps,
            resume_step,
            args.global_seed + train_res * 100,  # avoid same sampling for different resolutions
        )

        loader = DataLoader(
            dataset,
            batch_size=local_bsz,
            sampler=sampler,
            num_workers=args.num_workers,
            prefetch_factor=1,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=dataloader_collate_fn,
        )
        # default: 1000 steps, linear noise schedule
        transport = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=args.snr_type,
            mu=args.mu,
            do_shift=not args.no_shift,
            seq_len=(train_res // 16) ** 2,
        )  # default: velocity;

        data_collection[train_res] = {
            "loader": loader,
            "loader_iter": iter(loader),
            "global_bsz": global_bsz,
            "local_bsz": local_bsz,
            "micro_bsz": micro_bsz,
            "metrics": defaultdict(lambda: SmoothedValue(args.log_every)),
            "transport": transport,
        }

    # Prepare models for training:
    model.train()

    # Variables for monitoring/logging purposes:
    logger.info(f"Training for {args.max_steps:,} steps...")

    # 准备工作（提前上 GPU）
    dinov3_sp_mean = dinov3_sp_mean.permute(0, 2, 1).unsqueeze(-1).to(device).to(torch.bfloat16)
    dinov3_sp_std = dinov3_sp_std.permute(0, 2, 1).unsqueeze(-1).to(device).to(torch.bfloat16)

    for step in range(resume_step, args.max_steps):
        step_start_time = time()
        time_measurer.current_step = step

        for train_res, data_pack in data_collection.items():

            total_tokens = 0
            total_text_tokens = 0  # ✅ 新增：文本 token 总数统计

            # (1) 数据 prefetch 时间
            with time_measurer.measure_scope("data_prefetch", step):
                x, caps = next(data_pack["loader_iter"])
                x = [img.to(device, non_blocking=True) for img in x]

            # (2) DINO 预处理数据时间
            with time_measurer.measure_scope("dino_preprocessing", step):
                with torch.no_grad():
                    if args.single_shape:
                        x = torch.stack(x, dim=0)
                        x = dinov3.encode(x)  # (B', D, H', W')
                        x = (x - dinov3_sp_mean) / (dinov3_sp_std + 1e-8)
                    else:
                        # 按分辨率分组处理
                        size_to_idx = {}
                        for i, img in enumerate(x):
                            size_to_idx.setdefault(img.shape[-2:], []).append(i)

                        for (H, W), idxs in size_to_idx.items():
                            batch = torch.stack([x[i] for i in idxs], dim=0)
                            feats = dinov3.encode(batch)
                            B, D, Hf, Wf = feats.shape
                            total_tokens += Hf*Wf
                            feats = (feats - dinov3_sp_mean) / (dinov3_sp_std + 1e-8)
                            for i, fi in zip(idxs, feats):
                                x[i] = fi

            # 文本编码时间
            with time_measurer.measure_scope("text_encoding", step):
                with torch.no_grad():
                    cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob, args.max_cap_len)
                    # cap_mask: 1 表示有效 token，0 表示 padding
                    token_counts = cap_mask.sum(dim=1).float()  # shape [B]
                    # 当前 batch 的均值与方差
                    mean_tokens = token_counts.mean().item()
                    # var_tokens = token_counts.var(unbiased=False).item()  # 方差（总体）
                    std_tokens = token_counts.std(unbiased=False).item()  # 标准差（可选）

            loss_item = 0.0
            loss_256_item = 0.0
            loss_1024_item = 0.0

            opt.zero_grad()

            # 初始化计时变量
            total_forward_time = 0.0
            total_backward_time = 0.0

            # Number of bins, for loss recording
            n_loss_bins = 20
            # Create bins for t
            loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
            loss_bins_256 = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
            # Initialize occurrence and sum tensors
            bin_occurrence = torch.zeros(n_loss_bins, device="cuda")
            bin_occurrence_256 = torch.zeros(n_loss_bins, device="cuda")
            bin_sum_loss = torch.zeros(n_loss_bins, device="cuda")
            bin_sum_loss_256 = torch.zeros(n_loss_bins, device="cuda")

            for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
                mb_st = mb_idx * data_pack["micro_bsz"]
                mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
                last_mb = mb_ed == data_pack["local_bsz"]

                x_mb = x[mb_st:mb_ed]
                cap_feats_mb = cap_feats[mb_st:mb_ed]
                cap_mask_mb = cap_mask[mb_st:mb_ed]

                model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb)
                
                # (3) 模型 forward 时间
                with time_measurer.measure_scope("model_forward", step):
                    with {
                        "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                        "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                        "fp32": contextlib.nullcontext(),
                        "tf32": contextlib.nullcontext(),
                    }[args.precision]:
                        loss_dict = data_pack["transport"].training_losses(model, x_mb, model_kwargs)
                
                loss_256 = loss_dict["loss"].sum() / data_pack["local_bsz"]
                loss = loss_256
                loss_item += loss.item()
                loss_256_item += loss_256.item()

                # (4) 模型 backward 时间
                with time_measurer.measure_scope("model_backward", step):
                    with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                        loss.backward()

                # for bin-wise loss recording
                bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
                detached_loss = loss_dict["loss"].detach()

                for i in range(n_loss_bins):
                    mask = bin_indices == i
                    bin_occurrence[i] = bin_occurrence[i] + mask.sum()
                    bin_sum_loss[i] = bin_sum_loss[i] + detached_loss[mask].sum()

            # (5) 梯度裁剪时间
            with time_measurer.measure_scope("grad_clip", step):
                grad_norm = model.clip_grad_norm_(max_norm=args.grad_clip)

            dist.all_reduce(bin_occurrence)
            dist.all_reduce(bin_sum_loss)

            # (6) 模型参数 update 时间
            with time_measurer.measure_scope("optimizer_step", step):
                opt.step()

            # 计算其他边缘时间
            step_total_time = time() - step_start_time

            if tb_logger is not None:
                # 原有的指标记录
                tb_logger.add_scalar(f"{train_res}/loss", loss_item, step)
                tb_logger.add_scalar(f"{train_res}/loss_{train_res}", loss_256_item, step)
                tb_logger.add_scalar(f"{train_res}/grad_norm", grad_norm, step)
                tb_logger.add_scalar(f"{train_res}/lr", opt.param_groups[0]["lr"], step)
                tb_logger.add_scalar(f"timing/total_tokens", total_tokens, step)
                tb_logger.add_scalar(f"timing/total_text_tokens_mean", mean_tokens, step)
                tb_logger.add_scalar(f"timing/total_text_tokens_std", std_tokens, step)
                tb_logger.add_scalar(f"timing/Imgs-Sec", data_pack["global_bsz"] / step_total_time, step)
                
                for i in range(n_loss_bins):
                    if bin_occurrence[i] > 0:
                        bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                        tb_logger.add_scalar(f"{train_res}/loss-bin{i+1}-{n_loss_bins}", bin_avg_loss, step)

            # 替换为：
            if (step + 1) % 100 == 0 and rank == 0:
                time_measurer.print_time_distribution(logger, step + 1, data_pack["global_bsz"], step_total_time)

            # 原有的指标更新
            metrics = data_pack["metrics"]
            metrics["loss"].update(loss_item)
            metrics[f"loss_{train_res}"].update(loss_256_item)
            metrics["grad_norm"].update(grad_norm)
            metrics["Secs/Step"].update(step_total_time)
            metrics["Imgs/Sec"].update(data_pack["global_bsz"] / step_total_time)

            # === 每100秒统计一次图片吞吐量 ===
            if not hasattr(main, "_last_time"):
                main._last_time = time()
                main._img_counter = 0

            main._img_counter += data_pack["global_bsz"]
            elapsed = time() - main._last_time

            if elapsed >= 100.0:
                imgs_per_100s = main._img_counter / elapsed
                if rank == 0:
                    tb_logger.add_scalar("timing/imgs_per_100s", imgs_per_100s, step + 1)
                    logger.info(
                        f"[{step+1:06d}] Throughput: {imgs_per_100s:.2f} images/sec (averaged over last {elapsed:.1f}s)"
                    )
                main._img_counter = 0
                main._last_time = time()


            for i in range(n_loss_bins):
                if bin_occurrence[i] > 0:
                    bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                    metrics[f"bin_{train_res}_{i + 1:02}-{n_loss_bins}"].update(bin_avg_loss, int(bin_occurrence[i].item()))

            if (step + 1) % args.log_every == 0:
                torch.cuda.synchronize()
                logger.info(
                    f"Res{train_res}_{train_res//4}: (step{step + 1:07d}) "
                    + f"lr{opt.param_groups[0]['lr']:.6f} "
                    + " ".join([f"{key}:{str(metrics[key])}" for key in sorted(metrics.keys())])
                )

        # EMA 更新时间
        with time_measurer.measure_scope("ema_update", step):
            update_ema(model_ema, model)

        # Save DiT checkpoint:
        if step == 0 or (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT_Llama2_7B_patch2")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_bsz_1024", type=int, default=256)
    parser.add_argument("--micro_bsz_1024", type=int, default=1)
    parser.add_argument("--global_bsz_256", type=int, default=256)
    parser.add_argument("--micro_bsz_256", type=int, default=1)
    parser.add_argument("--global_bsz", type=int, default=256)
    parser.add_argument("--micro_bsz", type=int, default=1)
    parser.add_argument("--resol", type=int, default=1)
    parser.add_argument(
        "--use_long_cap",
        action="store_true",
    )
    parser.add_argument("--max_cap_len", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        default="",
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument(
        "--no_shift",
        action="store_true",
    )
    parser.add_argument(
        "--single_shape",
        action="store_true",
    )

    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--autoencoder_path", type=str, default=None)
    parser.add_argument("--gemma2b", type=str, default="google/gemma-2-2b")


    args = parser.parse_args()

    main(args)