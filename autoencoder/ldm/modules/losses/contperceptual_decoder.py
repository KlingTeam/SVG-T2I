import torch
import torch.nn as nn
from einops import rearrange

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminatorDecoder(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", distmat_margin=0, distmat_weight=1.0, cos_weight=1.0):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.distmat_weight = distmat_weight
        self.cos_weight = cos_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                    n_layers=disc_num_layers,
                                                    use_actnorm=use_actnorm
                                                    ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.distmat_margin = distmat_margin


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, student_features=None, teacher_features=None, last_layer=None, enc_last_layer=None, cond=None, split="train",
                weights=None, beta=2):

        # Handle the case where inputs and reconstructions are lists (process image by image)
        if isinstance(inputs, list):
            batch_rec_loss = 0.0
            batch_p_loss = 0.0
            batch_nll_loss = 0.0
            batch_size = len(inputs)
            
            for i, (input_img, recon_img) in enumerate(zip(inputs, reconstructions)):
                # Ensure the tensors are contiguous
                input_img = input_img.contiguous()
                recon_img = recon_img.contiguous()
                
                # Compute reconstruction loss
                rec_loss = torch.abs(input_img - recon_img)
                p_loss = torch.tensor(0.0, device=rec_loss.device)
                if self.perceptual_weight > 0:
                    p_loss = self.perceptual_loss(input_img, recon_img)
                    rec_loss = rec_loss + self.perceptual_weight * p_loss

                nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
                weighted_nll_loss = nll_loss
                if weights is not None:
                    weighted_nll_loss = weights*nll_loss
                
                rec_loss = torch.mean(rec_loss) / batch_size
                p_loss = torch.mean(p_loss) / batch_size
                weighted_nll_loss = torch.sum(weighted_nll_loss) / batch_size
                nll_loss = torch.sum(nll_loss) / batch_size
            
                batch_rec_loss += rec_loss
                batch_p_loss += p_loss
                batch_nll_loss += nll_loss

            # Compute averaged losses
            rec_loss = batch_rec_loss
            nll_loss = batch_nll_loss
            p_loss = batch_p_loss if self.perceptual_weight > 0 else torch.tensor(0.0)
            
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                rec_loss = rec_loss + self.perceptual_weight * p_loss

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # GAN part
        if optimizer_idx == 0:
            # Generator update
            if cond is None:
                assert not self.disc_conditional
                # Handle list case - compute discriminator output per image
                if isinstance(reconstructions, list):
                    logits_fake_list = []
                    for recon in reconstructions:
                        logits_fake = self.discriminator(recon.contiguous())
                        logits_fake_list.append(logits_fake)
                    # Average all discriminator outputs
                    logits_fake = torch.mean(torch.cat([lf.flatten() for lf in logits_fake_list]))               
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())                    
            else:
                assert self.disc_conditional
                # Handle list case
                if isinstance(reconstructions, list):
                    logits_fake_list = []
                    for recon, cond_img in zip(reconstructions, cond):
                        logits_fake = self.discriminator(torch.cat((recon.contiguous(), cond_img), dim=1))
                        logits_fake_list.append(logits_fake)
                    # Average all discriminator outputs
                    logits_fake = torch.mean(torch.cat([lf.flatten() for lf in logits_fake_list]))
                else:
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            
            g_loss = -torch.mean(logits_fake) if not isinstance(logits_fake, torch.Tensor) else -logits_fake
            

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
                
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            loss = weighted_nll_loss + d_weight * disc_factor * g_loss
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean() if self.perceptual_weight > 0 else torch.tensor(0.0),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            
            return loss, log

        if optimizer_idx == 1:
            # Discriminator update
            disc_losses = []
            logits_real_list = []
            logits_fake_list = []
            
            if cond is None:
                # Handle list case
                if isinstance(inputs, list):
                    for inp, recon in zip(inputs, reconstructions):
                        logits_real = self.discriminator(inp.contiguous().detach())
                        logits_fake = self.discriminator(recon.contiguous().detach())
                        current_d_loss = self.disc_loss(logits_real, logits_fake)
                        disc_losses.append(current_d_loss)
                        logits_real_list.append(logits_real.mean())
                        logits_fake_list.append(logits_fake.mean())
                else:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(reconstructions.contiguous().detach())
                    disc_losses.append(self.disc_loss(logits_real, logits_fake))
                    logits_real_list.append(logits_real.mean())
                    logits_fake_list.append(logits_fake.mean())
            else:
                # Handle list case
                if isinstance(inputs, list):
                    for inp, recon, cond_img in zip(inputs, reconstructions, cond):
                        logits_real = self.discriminator(torch.cat((inp.contiguous().detach(), cond_img), dim=1))
                        logits_fake = self.discriminator(torch.cat((recon.contiguous().detach(), cond_img), dim=1))
                        current_d_loss = self.disc_loss(logits_real, logits_fake)
                        disc_losses.append(current_d_loss)
                        logits_real_list.append(logits_real.mean())
                        logits_fake_list.append(logits_fake.mean())
                else:
                    logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                    logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
                    disc_losses.append(self.disc_loss(logits_real, logits_fake))
                    logits_real_list.append(logits_real.mean())
                    logits_fake_list.append(logits_fake.mean())

            # Compute averaged discriminator loss
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * torch.mean(torch.stack(disc_losses))

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): torch.mean(torch.stack(logits_real_list)),
                "{}/logits_fake".format(split): torch.mean(torch.stack(logits_fake_list))
            }
            return d_loss, log
