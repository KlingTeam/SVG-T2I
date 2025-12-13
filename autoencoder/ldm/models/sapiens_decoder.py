import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import torch.export
from PIL import Image
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.hy3.autoencoder_kl_3d import Decoder as HYDecoder
from ldm.util import instantiate_from_config
import sys

from ldm.sapiens_pretrain.mmpretrain.utils import register_all_modules
register_all_modules(init_default_scope=True)   
from ldm.sapiens_pretrain.mmpretrain.models.backbones.vision_transformer import VisionTransformer


class Sapiens_Decoder(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 sapiensconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 hyconfig=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_vf=None,
                 reverse_proj=False,
                 proj_fix=False,
                #  time_downsample=False,
                #  time_downsample_layer = None,
                 ):
        super().__init__()

        model_name = sapiensconfig.get('model_name', 'sapiens_0.3b')
        image_size = sapiensconfig.get('image_size', (1024, 768))
        patch_size = sapiensconfig.get('patch_size', 16)        

        # 2. 初始化模型
        model = VisionTransformer(
            arch=model_name,
            img_size=image_size,
            patch_size=patch_size,
            qkv_bias=True,
            final_norm=True,
            with_cls_token=False,   # Sapiens 特征提取通常不需要 CLS token
            out_type='featmap',     # 直接指定输出为特征图
            patch_cfg=dict(padding=2), # Sapiens 特有的 padding 设置
        )

        # 3. 加载权重
        checkpoint_path = sapiensconfig.get('checkpoint_path', None)
        
        print(f"Loading weights from: {checkpoint_path}")
        
        # 直接加载 .pt 文件
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # 处理权重字典的 key (移除 'backbone.' 或 'module.' 前缀)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果权重来自完整模型训练，通常带有 backbone. 前缀
            if k.startswith('backbone.'):
                new_state_dict[k[9:]] = v
            else:
                new_state_dict[k] = v

        # 加载处理后的权重
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Load status: {msg}")
        # 3. 冻结编码器权重
        self.encoder = model
        self.encoder.eval() # 设置为评估模式
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.use_hy = True if hyconfig is not None else False
        self.decoder = None
        self.image_key = 'image'
        if self.use_hy:
            self.decoder = HYDecoder(**hyconfig)
        for param in self.encoder.parameters():
            param.requires_grad = False
   
       
        self.loss = instantiate_from_config(lossconfig)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.reverse_proj = reverse_proj
        self.automatic_optimization = False
        self.proj_fix = proj_fix

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        x = (x + 1.0) / 2.0

        # 2. 进行 ImageNet 归一化
        # (x - mean) / std
        x = (x - self.mean) / self.std

        # 3. 前向传播
        # Sapiens VisionTransformer 直接接受 Tensor
        with torch.no_grad():
            # model(x) 返回一个 tuple，通常我们需要最后一层的特征
            features = self.encoder(x)
            
            # features 是一个 tuple，例如 (feat_stage_1, feat_stage_2, ..., feat_final)
            # 取最后一个元素
            h = features[-1]     
        # 为编码器增加时间维度
        h  = h.unsqueeze(2)
        return h
    
   
    
    def decode(self, z):

        dec = self.decoder(z)
        # print(f'dec.requires_grad: {dec.requires_grad}')
        return dec

    def forward(self, input):
        z = self.encode(input)
      
        dec = self.decode(z)

       
        return dec


    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # 只返回 [-1, 1] 范围的原始图像张量
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    def training_step(self, batch, batch_idx):

        # print(f'self.encoder.training: {self.encoder.training}')
        # print(f'self.decoder.training: {self.decoder.training}')
        # print(f'self.post_quant_conv.training: {self.post_quant_conv.training}')
        inputs = self.get_input(batch, self.image_key)
        reconstructions = self(inputs)
        reconstructions = reconstructions.view(reconstructions.shape[0],reconstructions.shape[1],reconstructions.shape[3],reconstructions.shape[4])
        ae_opt, disc_opt = self.optimizers()

        # if optimizer_idx == 0:
        # train encoder+decoder+logvar
        # inputs = inputs.unsqueeze(2)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, 0, self.global_step,
                                       last_layer=self.get_last_layer(),  split="train",  
                                       )
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        # return aeloss

        ae_opt.zero_grad()
        self.manual_backward(aeloss)
        ae_opt.step()

        # if optimizer_idx == 1:
        # train the discriminator
        # print(f'inputs.max(): {inputs.max()}, inputs.min(): {inputs.min()}')
        # import pdb; pdb.set_trace()
        discloss, log_dict_disc = self.loss(inputs, reconstructions, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", )

        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        # return discloss

        disc_opt.zero_grad()
        self.manual_backward(discloss)
        disc_opt.step()

    def validation_step(self, batch, batch_idx, dataloader_idx=0, data_type=None):
        inputs = self.get_input(batch, self.image_key)
        # print(f'Input shape is {inputs.shape}')
        reconstructions = self(inputs)
        reconstructions = reconstructions.view(reconstructions.shape[0],reconstructions.shape[1],reconstructions.shape[3],reconstructions.shape[4])
        print(f'Reconstructions shape is {reconstructions.shape}')
        print(f'Input shape is {inputs.shape}')
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, 0, self.global_step,
                                      last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions,  1, self.global_step,
                                         last_layer=self.get_last_layer(),   split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (list(self.decoder.parameters()) 
                #   + list(self.encoder.parameters())
                #   +list(self.post_quant_conv.parameters()) 
                  )
               
                
      
        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # x_dino = x_dino.to(self.device)
        if not only_inputs:
            xrec = self(x)
            
            # 处理 5D 输出 [B, C, T, H, W] -> [B, C, H, W]
            if xrec.dim() == 5:
                xrec = xrec.squeeze(2)

            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        print(f'X shape is {x.shape}')
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
