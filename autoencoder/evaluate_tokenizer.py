"""
Evaluate tokenizer performance by computing reconstruction metrics.

Metrics include:
- rFID (Reconstruction FID)
- PSNR (Peak Signal-to-Noise Ratio) 
- LPIPS (Learned Perceptual Image Patch Similarity)
- SSIM (Structural Similarity Index)

V1 by Jingfeng Yao from HUST-VL
V2 by Minglei Shi, Haolin Wang from THU
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision.datasets import ImageFolder
from torchvision import transforms
from diffusers.models import AutoencoderKL
from typing import Dict, List, Tuple, Optional, Callable
import logging
from omegaconf import OmegaConf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerEvaluator:
    """Main class for tokenizer evaluation."""
    
    def __init__(self, config_path: str, model_type: str, data_path: str, output_path: str, ckpt_path, ref_path=None):
        self.config_path = config_path
        self.model_type = model_type
        self.data_path = data_path
        self.output_path = output_path
        self.config_path = config_path
        self.config = OmegaConf.load(config_path)
        self.config.model.params.ckpt_path = ckpt_path

        # Initialize distributed training
        self._init_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Setup model and data
        self.model = self._load_model()
        self.transform = self._get_transform()
        self.dataset, self.dataloader = self._prepare_data()
        self.dataset_ref, self.dataloader_ref = self._prepare_data_ref()
        
        # Setup output directories
        self.save_dir, self.ref_path = self._setup_output_dirs()
        if ref_path is not None:
            self.ref_path = ref_path

        # Initialize metrics
        self.lpips = self._load_lpips().eval()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(self.device)
        
    def _init_distributed(self):
        """Initialize distributed training."""
        dist.init_process_group(backend='nccl')
        self.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(self.local_rank)
        
    def _load_model(self):
        """Load the specified model."""
        model_loaders = {
            'vavae': self._load_vavae,
            'dinov3': self._load_dinov3,
            'sdvae': self._load_sdvae,
            'marvae': self._load_marvae,
            'dinov2': self._load_dinov2,
            'siglip': self._load_siglip,
            'mae': self._load_mae,
            
        }
        
        if self.model_type not in model_loaders:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return model_loaders[self.model_type]()
    
    def _load_vavae(self):
        """Load VA-VAE model."""
        from tokenizer.vavae import VA_VAE
        model = VA_VAE(self.config_path).load().model
        if self.local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            encoder_params = sum(p.numel() for p in model.encoder.parameters())
            decoder_params = sum(p.numel() for p in model.decoder.parameters())
            
            logger.info(f"VA-VAE loaded with {total_params:,} total parameters")
            logger.info(f"Encoder parameters: {encoder_params:,}")
            logger.info(f"Decoder parameters: {decoder_params:,}")
            
        return model.to(self.device)
    
    def _load_dinov3(self):
        """Load DINOv3 model."""
        from ldm.util import instantiate_from_config
        model = instantiate_from_config(self.config.model).cuda().eval()

        return model.to(self.device)

    def _load_dinov2(self):
        config = OmegaConf.load(self.config_path)
        from ldm.models.dinov2_decoder import Dinov2Decoder
        model = Dinov2Decoder(
            ddconfig=config.model.params.ddconfig,
            dinoconfig=config.model.params.dinoconfig,
            lossconfig=config.model.params.lossconfig,
            embed_dim=config.model.params.embed_dim,
            ckpt_path=self.config.ckpt_path
        ).cuda().eval()

        return model.to(self.device)

    def _load_mae(self):
        config = OmegaConf.load(self.config_path)
        from ldm.models.mae_decoder import MaeDecoder
        model = MaeDecoder(
            ddconfig=config.model.params.ddconfig,
            maeconfig=config.model.params.maeconfig,
            lossconfig=config.model.params.lossconfig,
            embed_dim=config.model.params.embed_dim,
            ckpt_path=self.config.ckpt_path
        ).cuda().eval()
        return model.to(self.device)

    def _load_siglip(self):
        config = OmegaConf.load(self.config_path)
        from ldm.models.siglip2_decoder import SigLipv2Decoder
        model = SigLipv2Decoder(
            ddconfig=config.model.params.ddconfig,
            siglipconfig=config.model.params.siglipconfig,
            lossconfig=config.model.params.lossconfig,
            embed_dim=config.model.params.embed_dim,
            ckpt_path=self.config.ckpt_path
        ).cuda().eval()
        return model.to(self.device)


    def _load_sdvae(self):
        """Load Stable Diffusion VAE."""
        # Note: Update the path to your actual SD-VAE model
        return AutoencoderKL.from_pretrained("path/to/your/sd-vae-ft-ema").to(self.device)
    
    def _load_marvae(self):
        """Load MAR-VAE model."""
        from tokenizer.marvae import MAR_VAE
        return MAR_VAE().load().model.to(self.device)
    
    def _load_lpips(self):
        """Load LPIPS model."""
        from models.lpips import LPIPS
        return LPIPS().to(self.device)
    
    def _get_transform(self):
        """Get appropriate image transformation."""
        if self.model_type == 'dinov3' or self.model_type == 'dinov2' or self.model_type == 'mae' or self.model_type == 'siglip':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def _prepare_data(self):
        """Prepare dataset and dataloader."""
        dataset = ImageFolder(root=self.data_path, transform=self.transform)
        sampler = DistributedSampler(
            dataset, 
            num_replicas=dist.get_world_size(), 
            rank=self.local_rank
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=200,
            shuffle=False,
            num_workers=4,
            sampler=sampler
        )
        
        return dataset, dataloader

    def _prepare_data_ref(self):
        """Prepare dataset and dataloader."""
        dataset = ImageFolder(root=self.data_path, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]))
        sampler = DistributedSampler(
            dataset, 
            num_replicas=dist.get_world_size(), 
            rank=self.local_rank
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=200,
            shuffle=False,
            num_workers=4,
            sampler=sampler
        )
        
        return dataset, dataloader

    def _setup_output_dirs(self):
        """Setup output directories for results."""
        ckpt_epoch = ".".join(self.config.model.params.ckpt_path.split('/')[-1].split('.')[:-1])
        folder_names = {
            'vavae': os.path.splitext(os.path.basename(self.config_path))[0],
            'sdvae': 'sdvae',
            'marvae': 'marvae',
            'dinov3': os.path.splitext(os.path.basename(self.config_path))[0] + "_" + ckpt_epoch,
            'dinov2': os.path.splitext(os.path.basename(self.config_path))[0] + "_" + ckpt_epoch,
            'siglip': os.path.splitext(os.path.basename(self.config_path))[0] + "_" + ckpt_epoch,
            'mae': os.path.splitext(os.path.basename(self.config_path))[0] + "_" + ckpt_epoch,
        }


        folder_name = folder_names[self.model_type]
        base_dir = os.path.join(self.output_path, folder_name)
        self.base_dir = base_dir
        save_dir = os.path.join(base_dir, 'decoded_images')
        ref_path = os.path.join(base_dir, 'ref_images')
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(ref_path, exist_ok=True)
        
        if self.local_rank == 0:
            logger.info(f"Output dir: {save_dir}")
            logger.info(f"Reference dir: {ref_path}")
            
        return save_dir, ref_path
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space."""
        # import ipdb; ipdb.set_trace()
        encode_methods = {
            'vavae': lambda: self.model.encode(images),
            'marvae': lambda: self.model.encode(images),
            'sdvae': lambda: self.model.encode(images).latent_dist,
            'dinov3': lambda: self.model.encode(images),
            'dinov2': lambda: self.model.encode(images),
            'siglip': lambda: self.model.encode(images),
            'mae': lambda: self.model.encode(images),
        }
        
        with torch.no_grad():
            posterior = encode_methods[self.model_type]()
            
            if self.model_type == 'dinov3' or self.model_type == 'dinov2' or self.model_type == 'mae' or self.model_type == 'siglip':
                return posterior
                
            return posterior.sample().to(torch.float32)
    
    def decode_images(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to images."""
        decode_methods = {
            'vavae': lambda: self.model.decode(latents),
            'marvae': lambda: self.model.decode(latents),
            'sdvae': lambda: self.model.decode(latents).sample,
            'dinov3': lambda: self.model.decode(latents),
            'dinov2': lambda: self.model.decode(latents),
            'siglip': lambda: self.model.decode(latents),
            'mae': lambda: self.model.decode(latents),
        }
        
        with torch.no_grad():
            return decode_methods[self.model_type]()
    
    def save_reference_images(self):
        """Save reference images if needed."""
        ref_png_files = [f for f in os.listdir(self.ref_path) if f.endswith('.png')]
        
        if len(ref_png_files) < 50000:
            total_samples = 0
            for batch in self.dataloader_ref:
                images = batch[0].to(self.device)
                for j in range(images.size(0)):
                    img = torch.clamp(127.5 * images[j] + 128.0, 0, 255)
                    img = img.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                    
                    Image.fromarray(img).save(
                        os.path.join(self.ref_path, f"ref_image_rank_{self.local_rank}_{total_samples}.png")
                    )
                    
                    total_samples += 1
                    if total_samples % 100 == 0 and self.local_rank == 0:
                        logger.info(f"Rank {self.local_rank}, Saved {total_samples} reference images")
            
            dist.barrier()

    # Add validation before calculating SSIM
    def calculate_ssim(self, pred, target):
        # Ensure shapes match
        if pred.shape != target.shape:
            logger.warning(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
            # Try resizing to match shapes
            pred = torch.nn.functional.interpolate(
                pred, size=target.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Ensure values are within expected range
        pred_min, pred_max = pred.min(), pred.max()
        target_min, target_max = target.min(), target.max()
        if not (torch.isclose(torch.tensor(pred_min), torch.tensor(-1.0), atol=0.1) and 
                torch.isclose(torch.tensor(pred_max), torch.tensor(1.0), atol=0.1)):
            logger.warning(f"Prediction out of range: min {pred_min:.2f}, max {pred_max:.2f}")
        
        return self.ssim_metric(pred, target)


    def evaluate(self):
        """Run the evaluation pipeline."""
        # Save reference images if needed
        self.save_reference_images()
        # Generate reconstructions and compute metrics
        lpips_values = []
        ssim_values = []
        all_indices = 0
        
        if self.local_rank == 0:
            logger.info("Generating reconstructions...")
        
        for batch in tqdm(self.dataloader_ref, disable=self.local_rank != 0):
            images = batch[0].to(self.device)
            if self.model_type == 'dinov3' or self.model_type == 'dinov2' or self.model_type == 'mae' or self.model_type == 'siglip':
                tmp = (batch[0] + 1.0) / 2.0
                if self.model_type == 'dinov2' or self.model_type == 'mae':
                    tmp = transforms.Resize((224, 224))(tmp)
                
                tmp = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tmp)
                images_input = tmp.to(self.device)
            else:
                images_input = images
            
            # Encode and decode images
            latents = self.encode_images(images_input)
            decoded_images_tensor = self.decode_images(latents)

            decoded_images = torch.clamp(127.5 * decoded_images_tensor + 128.0, 0, 255)
            decoded_images_tensor_norm = decoded_images / 255.0 * 2 - 1
            decoded_images = decoded_images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            # Compute metrics
            lpips_values.append(self.lpips(decoded_images_tensor_norm, images).mean())
            ssim_values.append(self.ssim_metric(decoded_images_tensor_norm, images))
            # ssim_values.append(self.calculate_ssim(decoded_images_tensor_norm, images))

            print(f'lpips: {lpips_values[-1].item()}, ssim: {ssim_values[-1].item()}')
            # Save reconstructions
            for i, img in enumerate(decoded_images):
                Image.fromarray(img).save(
                    os.path.join(self.save_dir, f"decoded_image_rank_{self.local_rank}_{all_indices + i}.png")
                )
                
            all_indices += len(decoded_images)
        
        # Aggregate metrics across GPUs
        lpips_values = torch.tensor(lpips_values).to(self.device)
        ssim_values = torch.tensor(ssim_values).to(self.device)
        
        dist.all_reduce(lpips_values, op=dist.ReduceOp.AVG)
        dist.all_reduce(ssim_values, op=dist.ReduceOp.AVG)
        
        avg_lpips = lpips_values.mean().item()
        avg_ssim = ssim_values.mean().item()
        
        # Calculate additional metrics on rank 0
        if self.local_rank == 0:
            from tools.calculate_fid import calculate_fid_given_paths
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Calculate FID
            logger.info("Computing rFID...")
            fid = calculate_fid_given_paths(
                [self.ref_path, self.save_dir], 
                batch_size=50, 
                dims=2048, 
                device=self.device, 
                num_workers=16
            )
            
            # Calculate PSNR
            logger.info("Computing PSNR...")
            psnr_values = self.calculate_psnr_between_folders(self.ref_path, self.save_dir)
            avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
            
            # Print final results
            logger.info("Final Metrics:")
            logger.info(f"rFID: {fid:.3f}")
            logger.info(f"PSNR: {avg_psnr:.3f}")
            logger.info(f"LPIPS: {avg_lpips:.3f}")
            logger.info(f"SSIM: {avg_ssim:.3f}")
        
            import csv
            save_path = os.path.join(self.base_dir, "metrics.csv")
            file_exists = os.path.isfile(save_path)

            with open(save_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["rFID", "PSNR", "LPIPS", "SSIM"])
                writer.writerow([round(fid, 3), round(avg_psnr, 3), round(avg_lpips, 3), round(avg_ssim, 3)])

            logger.info(f"Metrics saved to {save_path}")

        dist.destroy_process_group()
    
    @staticmethod
    def calculate_psnr(original: torch.Tensor, processed: torch.Tensor) -> float:
        """Calculate PSNR between two images."""
        mse = torch.mean((original - processed) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()
    
    @staticmethod
    def load_image(image_path: str) -> torch.Tensor:
        """Load an image as a tensor."""
        image = Image.open(image_path).convert('RGB')
        return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)
    
    def calculate_psnr_for_pair(self, original_path: str, processed_path: str) -> float:
        """Calculate PSNR for a pair of images."""
        return self.calculate_psnr(
            self.load_image(original_path), 
            self.load_image(processed_path)
        )
    
    def calculate_psnr_between_folders(self, original_folder: str, processed_folder: str) -> List[float]:
        """Calculate PSNR between all images in two folders."""
        original_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.png')])
        processed_files = sorted([f for f in os.listdir(processed_folder) if f.endswith('.png')])
        
        if len(original_files) != len(processed_files):
            logger.warning("Mismatched number of images in folders")
            return []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        psnr_values = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.calculate_psnr_for_pair,
                    os.path.join(original_folder, orig),
                    os.path.join(processed_folder, proc)
                )
                for orig, proc in zip(original_files, processed_files)
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating PSNR"):
                psnr_values.append(future.result())
                
        return psnr_values


def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate tokenizer performance")
    parser.add_argument('--config_path', type=str, default='/path/autoencoder')
    parser.add_argument('--model_type', type=str, default='dinov3')
    parser.add_argument('--data_path', type=str, default='/path/to/your/imagenet/ILSVRC2012_validation/data')
    parser.add_argument('--ckpt_path', type=str, default='/path/to/your/imagenet/ILSVRC2012_validation/data')
    parser.add_argument('--ref_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='/path/to/your/output')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run evaluation
    evaluator = TokenizerEvaluator(
        config_path=args.config_path,
        model_type=args.model_type,
        data_path=args.data_path,
        output_path=args.output_path,
        ckpt_path=args.ckpt_path,
        ref_path=args.ref_path,
    )
    
    evaluator.evaluate()


if __name__ == "__main__":
    main()