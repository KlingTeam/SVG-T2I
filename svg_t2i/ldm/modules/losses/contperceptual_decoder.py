import torch
import torch.nn as nn
from einops import rearrange

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminatorDecoder(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", distill_factor=0.0,
                 distmat_margin=0, distmat_weight=1.0, cos_weight=1.0):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.distill_factor = distill_factor
        
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

    def calculate_adaptive_weight_distill(self, nll_loss, distill_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            distill_grads = torch.autograd.grad(distill_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            distill_grads = torch.autograd.grad(distill_loss, self.last_layer[0], retain_graph=True)[0]

        distill_weight = torch.norm(nll_grads) / (torch.norm(distill_grads) + 1e-4)
        distill_weight = torch.clamp(distill_weight, 0.0, 1e8).detach()
        distill_weight = distill_weight * self.distill_factor
        return distill_weight


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

        # 处理inputs和reconstructions为list的情况（逐图处理）
        if isinstance(inputs, list):
            batch_rec_loss = 0.0
            batch_p_loss = 0.0
            batch_nll_loss = 0.0
            batch_size = len(inputs)
            
            for i, (input_img, recon_img) in enumerate(zip(inputs, reconstructions)):
                # 确保图像是连续的
                input_img = input_img.contiguous()
                recon_img = recon_img.contiguous()
                
                
                # 计算重建损失
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

            # 计算平均损失
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


        # GAN部分
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                # 处理list情况 - 分别计算每个图像的判别器输出
                if isinstance(reconstructions, list):
                    logits_fake_list = []
                    for recon in reconstructions:
                        logits_fake = self.discriminator(recon.contiguous())
                        logits_fake_list.append(logits_fake)
                    # 平均所有图像的判别器输出
                    logits_fake = torch.mean(torch.cat([lf.flatten() for lf in logits_fake_list]))               
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())                    
            else:
                assert self.disc_conditional
                # 处理list情况
                if isinstance(reconstructions, list):
                    logits_fake_list = []
                    for recon, cond_img in zip(reconstructions, cond):
                        logits_fake = self.discriminator(torch.cat((recon.contiguous(), cond_img), dim=1))
                        logits_fake_list.append(logits_fake)
                    # 平均所有图像的判别器输出
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


            if self.distill_factor > 0.0 and student_features is not None and teacher_features is not None:
                if isinstance(student_features, list):
                    distillation_loss = 0.0
                    batch_num = len(student_features)
                    for sf, tf in zip(student_features, teacher_features):
                        num_layers = len(sf)
                        device = sf[0].device
                        per_layer_cosine_distances = []
                        for student_feat, teacher_feat in zip(sf, tf):
                            similarity = F.cosine_similarity(student_feat, teacher_feat.detach(), dim=-1).mean(dim=1)  
                            distance = 1 - similarity
                            per_layer_cosine_distances.append(distance)
                        alignment_penalty = torch.stack(per_layer_cosine_distances)
                        hierarchical_prior = torch.arange(1, num_layers + 1, device=device) / num_layers
                        log_hierarchical_prior = torch.log(hierarchical_prior + 1e-8).unsqueeze(1) 
                        scores = log_hierarchical_prior + alignment_penalty.detach() * beta
                        adaptive_weights = F.softmax(scores, dim=0)
                        distillation_loss += torch.sum(adaptive_weights * alignment_penalty, dim = 0)
                        
                    distillation_loss = distillation_loss / batch_num
                else:
                    num_layers = len(student_features)
                    # print(f'num_layers for distillation: {num_layers}')
                    device = student_features[0].device
                    per_layer_cosine_distances = []
                    for student_feat, teacher_feat in zip(student_features, teacher_features):
                        # We want the mean similarity across all tokens in the batch for this layer
                        # print(f'Student feat shape is {student_feat.shape}')
                        similarity = F.cosine_similarity(student_feat, teacher_feat.detach(), dim=-1).mean(dim=1)  
                        distance = 1 - similarity
                        per_layer_cosine_distances.append(distance)
                    
                    alignment_penalty = torch.stack(per_layer_cosine_distances)
                    hierarchical_prior = torch.arange(1, num_layers + 1, device=device) / num_layers
                    log_hierarchical_prior = torch.log(hierarchical_prior + 1e-8).unsqueeze(1)  # add small constant for numerical stability
                    # Combine with alignment penalty and softmax
                    # Use .detach() on the penalty so weights are not part of the backprop graph themselves
                    scores = log_hierarchical_prior + alignment_penalty.detach() * beta
                    adaptive_weights = F.softmax(scores, dim=0)

                    # Step 3: Calculate final weighted distillation loss (L_dist)
                    distillation_loss = torch.sum(adaptive_weights * alignment_penalty, dim = 0)
            
                if split == 'train':
                    distill_weight = self.calculate_adaptive_weight_distill(nll_loss, distillation_loss, last_layer=enc_last_layer)
                else:
                    distill_weight = 1.0
                    
                loss = weighted_nll_loss + d_weight * disc_factor * g_loss + distill_weight * distillation_loss
                log = {
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/p_loss".format(split): p_loss.detach().mean() if self.perceptual_weight > 0 else torch.tensor(0.0),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                    "{}/distill_loss".format(split): distillation_loss.detach().mean(),
                    "{}/distill_weight".format(split): torch.tensor(distill_weight).detach(),
                }
            else:
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
            # discriminator update
            disc_losses = []
            logits_real_list = []
            logits_fake_list = []
            
            if cond is None:
                # 处理list情况
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
                # 处理list情况
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

            # 计算平均判别器损失
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * torch.mean(torch.stack(disc_losses))

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): torch.mean(torch.stack(logits_real_list)),
                "{}/logits_fake".format(split): torch.mean(torch.stack(logits_fake_list))
            }
            return d_loss, log

    def forward_batch(self, inputs, reconstructions, optimizer_idx,
                global_step, student_features=None, teacher_features=None, last_layer=None, enc_last_layer=None, cond=None, split="train",
                weights=None, beta=2):
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


        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            # d_weight = 2000.0
            # d_weight = 1.0
            
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            

            distillation_loss = torch.tensor(0.0, device=inputs.device)
            if self.distill_factor > 0.0 and student_features is not None and teacher_features is not None:
                num_layers = len(student_features)
                # print(f'num_layers for distillation: {num_layers}')
                device = student_features[0].device
                per_layer_cosine_distances = []
                for student_feat, teacher_feat in zip(student_features, teacher_features):
                    # We want the mean similarity across all tokens in the batch for this layer
                    # print(f'Student feat shape is {student_feat.shape}')
                    similarity = F.cosine_similarity(student_feat, teacher_feat.detach(), dim=-1).mean(dim=1)  
                    distance = 1 - similarity
                    per_layer_cosine_distances.append(distance)
                
                alignment_penalty = torch.stack(per_layer_cosine_distances)
                hierarchical_prior = torch.arange(1, num_layers + 1, device=device) / num_layers
                log_hierarchical_prior = torch.log(hierarchical_prior + 1e-8).unsqueeze(1)  # add small constant for numerical stability
                # Combine with alignment penalty and softmax
                # Use .detach() on the penalty so weights are not part of the backprop graph themselves
                scores = log_hierarchical_prior + alignment_penalty.detach() * beta
                adaptive_weights = F.softmax(scores, dim=0)

                # Step 3: Calculate final weighted distillation loss (L_dist)
                distillation_loss = torch.sum(adaptive_weights * alignment_penalty, dim = 0)
                distillation_loss = torch.mean(distillation_loss)
                if split == 'train':
                    distill_weight = self.calculate_adaptive_weight_distill(nll_loss, distillation_loss, last_layer=enc_last_layer)
                else:
                    distill_weight = 1.0
                loss = weighted_nll_loss  +  d_weight * disc_factor * g_loss + distill_weight * distillation_loss
            else:
                loss = weighted_nll_loss  +  d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean() if self.perceptual_weight > 0 else torch.tensor(0.0),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(self.disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/distill_loss".format(split): distillation_loss.detach().mean(),
                   "{}/distill_weight".format(split): torch.tensor(distill_weight).detach(),
                   }
           
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

