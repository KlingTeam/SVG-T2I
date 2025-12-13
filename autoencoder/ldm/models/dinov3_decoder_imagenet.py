import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ldm.modules.diffusionmodules.model import Decoder
from ldm.hy3.autoencoder_kl_3d import Decoder as HYDecoder
from ldm.util import instantiate_from_config
from torchvision.models.vision_transformer import VisionTransformer
from ldm.models.swin_v2 import SwinV2Encoder
from ldm.rope_vit import vit_rope


def create_small_vit_s(output_dim=8, patch_size=16, img_size=256, hidden_dim=384, vit_type='vit-s'):
    """
    Create a lightweight ViT-S model.

    Args:
        output_dim: Output feature dimension.
        patch_size: Patch size for input images.
        img_size: Input image size.
    """

    if vit_type=='vit-s':
        # Small ViT-S configuration
        vit_config = {
            'image_size': img_size,
            'patch_size': patch_size,
            'num_layers': 6,         # fewer layers for a lightweight model
            'num_heads': 8,          # fewer attention heads
            'hidden_dim': hidden_dim,       # smaller hidden dimension
            'mlp_dim': 1536,         # typically 4x hidden_dim
            'num_classes': output_dim,
            'dropout': 0.1,
            'attention_dropout': 0.1,
        }
        model = VisionTransformer(**vit_config)
    elif vit_type=='vit-rope-s':
        model = vit_rope.rope_mixed_deit_small_patch16_LS(pretrained=False, img_size=1024)
    elif vit_type=='swin-t':
        model = SwinV2Encoder(
                        variant="t",
                        out_channels=8,
                        head="skip",            # try 'mlp' with mlp_stage_index=2 as well
                        mlp_stage_index=2,
                        in_channels=3,
                        input_resolution=None,
                        window_size=8,
                        patch_size=4,
                        use_checkpoint=False,
                        sequential_self_attention=False,
                        use_deformable_block=False,
                    )
    # Replace classification head with a linear projection to output_dim
    # The output shape will be (B, 8, 256)
    if vit_type != 'swin-t':
        model.heads = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Small ViT Total parameters: {total_params:,}")
    print(f"Small ViT Trainable parameters: {trainable_params:,}")

    # Define custom forward to return patch-level features
    if vit_type != 'swin-t':

        def forward_custom(x):
            # Extract features via ViT
            if vit_type == 'vit-s':

                x = model._process_input(x)
                n = x.shape[1]

                # Add class token
                batch_size = x.shape[0]
                cls_tokens = model.class_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

                # Pass through Transformer encoder
                x = model.encoder(x)

                # Remove class token, keep patch tokens only
                x = x[:, 1:, :]  # shape: (B, 256, 384)

                # Apply head projection
                x = model.heads(x)  # shape: (B, 256, 8)

                # Transpose to (B, 8, 256)
                return x.transpose(1, 2)

            elif vit_type == 'vit-rope-s':
                x = model.forward_features(x)

                # Remove class token, keep patch tokens only
                x = x[:, 1:, :]  # shape: (B, 256, 384)

                # Apply head projection
                x = model.heads(x)  # shape: (B, 256, 8)

                # Transpose to (B, 8, 256)
                return x.transpose(1, 2)

        model.forward = forward_custom

    return model

def match_distribution(h, h_vit, eps=1e-6):
    """
    Match h_vit distribution to h distribution.

    Args:
        h: [B, D1, N]   (DINO features)
        h_vit: [B, D2, N] (ViT features)
    """
    # Compute global mean and std for DINO features
    mean_h = h.mean(dim=(0, 2), keepdim=True)
    std_h = h.std(dim=(0, 2), keepdim=True)

    mean_h_scalar = mean_h.mean().detach()
    std_h_scalar = std_h.mean().detach()

    # Compute mean and std for ViT features
    mean_vit = h_vit.mean(dim=(0, 2), keepdim=True)
    std_vit = h_vit.std(dim=(0, 2), keepdim=True)

    mean_vit_scalar = mean_vit.mean().detach()
    std_vit_scalar = std_vit.mean().detach()

    # Normalize and re-scale
    h_vit_normed = (h_vit - mean_vit_scalar) / (std_vit_scalar + eps)
    h_vit_aligned = h_vit_normed * std_h_scalar + mean_h_scalar

    return h_vit_aligned

class DinoDecoder(pl.LightningModule):
    def __init__(self,
                dinoconfig,
                lossconfig,
                embed_dim,
                extra_vit_config=None,
                hyconfig=None,
                ddconfig=None,
                ckpt_path=None,
                ignore_keys=None,
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                proj_fix=False,
                is_train=True):
        super().__init__()
        ignore_keys = ignore_keys or []
        self.image_key = image_key

        self.use_hy = hyconfig is not None
        # Load DINO encoder
        self.encoder = torch.hub.load(
            repo_or_dir=dinoconfig['dinov3_location'],
            model=dinoconfig['model_name'],
            source="local",
            weights=dinoconfig['weights'],
        ).eval()

        # Extra lightweight ViT
        self.use_extra_vit = extra_vit_config is not None
        print(extra_vit_config)
        if self.use_extra_vit:
            self.extra_vit = create_small_vit_s(output_dim=extra_vit_config['output_dim'], vit_type=extra_vit_config['vit_type'])

            self.mask_ratio = extra_vit_config.get('mask_ratio', 0.0)
            self.use_outnorm = extra_vit_config.get('use_outnorm', False)
            self.frozen_vit = extra_vit_config.get('frozen', False)
            if self.frozen_vit:
                self.extra_vit.eval()

            if self.mask_ratio > 0:
                self.mask_token = nn.Parameter(torch.zeros(1, extra_vit_config['output_dim'], 1))
                nn.init.normal_(self.mask_token, std=0.02)

            self.norm_vit = nn.LayerNorm(extra_vit_config['output_dim'] + embed_dim)

        # Decoder
        self.decoder = HYDecoder(**hyconfig) if self.use_hy else Decoder(**ddconfig)

        # Loss
        self.loss = instantiate_from_config(lossconfig) if is_train else None

        # Optional visualization
        if colorize_nlabels is not None:
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

        self.automatic_optimization = False

    def init_from_ckpt(self, path, ignore_keys):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
            # if "extra_vit" not in k:
                sd.pop(k)
        self.load_state_dict(sd, strict=False)
        # import ipdb;ipdb.set_trace()
        # self.load_state_dict(sd)
        print(f"Restored from {path}")

    def encode(self, x):
        """Encode images to latent features with optional extra ViT."""
        h = self.encoder.forward_features(x)['x_norm_patchtokens']  # [B, D, N]
        # h = self.encoder.forward_patch_only(x)  # [B, D, N]
        h = h.permute(0, 2, 1)  # [B, N, D]
        # print(self.use_extra_vit)
        if self.use_extra_vit:
            h_vit = self.extra_vit(x)  # [B, D2, N]
            # if self.training and self.mask_ratio > 0:
                # B, D, N = h_vit.shape
                # mask_flags = (torch.rand(B, device=x.device) < self.mask_ratio).view(B, 1, 1)
                # h_vit = h_vit * (1 - mask_flags) + self.mask_token.expand(B, D, N) * mask_flags
            if self.use_outnorm:
                h_vit = match_distribution(h, h_vit)
            h = torch.cat([h, h_vit], dim=1)

        # reshape to [B, D_total, H_patch, W_patch]
        h = h.view(h.shape[0], -1, int(x.shape[2] // 16), int(x.shape[3] // 16)).contiguous()
        return h.contiguous()

    def decode(self, z):
        return self.decoder(z.unsqueeze(2)).squeeze(2) if self.use_hy else self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def get_input(self, batch, key):
        x = batch[key]
        if x.ndim == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous().float()
        x_dino = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_dino = (x_dino - mean) / std
        return x, x_dino

    def training_step(self, batch, batch_idx):
        x, x_dino = self.get_input(batch, self.image_key)
        recon = self(x_dino)
        ae_opt, disc_opt = self.optimizers()

        # Autoencoder update
        aeloss, log_ae = self.loss(x, recon, 0, self.global_step,
                                   last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True)
        self.log_dict(log_ae, prog_bar=False)
        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        # Discriminator update
        discloss, log_disc = self.loss(x, recon, 1, self.global_step,
                                       last_layer=self.get_last_layer(), split="train")
        self.log("discloss", discloss, prog_bar=True, logger=True)
        self.log_dict(log_disc, prog_bar=False)

        actual_step = int(self.global_step // 2)
        # print(actual_step)
        self.log("actual_step", actual_step, on_step=True, on_epoch=False)

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

    def validation_step(self, batch, batch_idx):
        x, x_dino = self.get_input(batch, self.image_key)
        recon = self(x_dino)
        aeloss, log_ae = self.loss(x, recon, 0, self.global_step,
                                   last_layer=self.get_last_layer(), split="val")
        discloss, log_disc = self.loss(x, recon, 1, self.global_step,
                                       last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss)
        self.log("val/discloss", discloss)
        self.log_dict(log_ae)
        self.log_dict(log_disc)

    def configure_optimizers(self):
        params = list(self.decoder.parameters())
        if self.use_extra_vit and not self.frozen_vit:
            params += list(self.extra_vit.parameters())
            if self.mask_ratio > 0:
                params.append(self.mask_token)
        opt_ae = torch.optim.Adam(params, lr=self.learning_rate, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """Return the final decoder layer for logging losses."""
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = {}
        x, x_dino = self.get_input(batch, self.image_key)
        x, x_dino = x.to(self.device), x_dino.to(self.device)
        if not only_inputs:
            recon = self(x_dino)
            if x.shape[1] > 3:
                x = self.to_rgb(x)
                recon = self.to_rgb(recon)

            log["reconstructions"] = recon
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        return 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
