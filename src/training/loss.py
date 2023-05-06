# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""
from typing import Tuple, Optional
import numpy as np
import ot
import torch
import torch.nn.functional as F
from src.dnnlib import TensorGroup, EasyDict
from src.torch_utils import training_stats
from src.torch_utils import misc
from src.torch_utils.ops import conv2d_gradfix
from src.torch_utils.ops import upfirdn2d
from src.training.training_utils import sample_patch_params, extract_patches, linear_schedule, sample_random_c
from src.training.rendering_utils import get_mean_angles_values

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, *args, **kwargs): # to be overridden by subclass
        raise NotImplementedError()

    def progressive_update(self, *args, **kwargs):
        pass

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, cfg, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_batch_shrink=2, pl_decay=0.01):
        super().__init__()
        self.cfg                = cfg
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = cfg.model.loss_kwargs.get('blur_init_sigma', 0)
        self.blur_fade_kimg     = cfg.model.loss_kwargs.get('blur_fade_kimg', 0)
        self.patch_cfg          = EasyDict.init_recursively(self.cfg.training.patch) # For faster field access

        self.progressive_update(0)

    def progressive_update(self, cur_kimg: int):
        if self.patch_cfg.enabled:
            if self.patch_cfg.distribution in ('uniform', 'discrete_uniform'):
                self.patch_cfg.min_scale = linear_schedule(cur_kimg, self.patch_cfg.max_scale, self.patch_cfg.min_scale_trg, self.patch_cfg.anneal_kimg)
            elif self.patch_cfg.distribution == 'beta':
                self.patch_cfg.beta = linear_schedule(cur_kimg, self.patch_cfg.beta_val_start, self.patch_cfg.beta_val_end, self.patch_cfg.anneal_kimg)
                self.patch_cfg.min_scale = self.patch_cfg.min_scale_trg
            else:
                raise NotImplementedError(f"Uknown patch distribution: {self.patch_cfg.distribution}")
        self.gpc_spoof_p = linear_schedule(cur_kimg, 1.0, self.cfg.model.generator.camera_cond_spoof_p, 1000)
        self.D_kd_weight = linear_schedule(cur_kimg, self.cfg.model.loss_kwargs.kd.discr.weight, 0.0, period=self.cfg.model.loss_kwargs.kd.discr.anneal_kimg, start_step=0)
        if self.cfg.training.learn_camera_dist:
            self.emd_multiplier = linear_schedule(cur_kimg, 0.0, 1.0, period=self.cfg.model.generator.camera_adaptor.emd.anneal_kimg, start_step=0)
        else:
            self.emd_multiplier = 0.0

    def run_G(self, z, c, camera_params, camera_angles_cond=None, update_emas=False):
        ws = self.G.mapping(z=z, c=c, camera_angles=camera_angles_cond, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(z=torch.randn_like(z), c=c, camera_angles=camera_angles_cond, update_emas=False)[:, cutoff:]
        patch_params = sample_patch_params(len(z), self.patch_cfg, device=z.device) if self.patch_cfg.enabled else {}
        patch_kwargs = dict(patch_params=patch_params) if self.patch_cfg.enabled else {}
        if self.cfg.training.learn_camera_dist:
            camera_params = self.G.synthesis.camera_adaptor(camera_params, z, c)
        render_opts_overwrites = dict(concat_depth=self.cfg.training.use_depth, return_depth=True)
        out = self.G.synthesis(ws, camera_params, update_emas=update_emas, render_opts=render_opts_overwrites, **patch_kwargs)
        out.ws = ws
        return out, patch_params, camera_params

    def run_D(self, img, c, blur_sigma=0, update_emas=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img = maybe_blur(img, blur_sigma) # [batch_size, c, h, w]

        assert img.shape[1] == 4 or not self.cfg.training.use_depth, f"Wrong shape: {img.shape}"

        if self.cfg.training.use_depth:
            with torch.autograd.profiler.record_function('depth_blur'):
                blur_size = np.floor(blur_sigma * 3)
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(30.0).square().neg().exp2()
                depth_blurred = upfirdn2d.filter2d(img[:, [3]], f / f.sum()) # [batch_size, 1, h, w]
                img = torch.cat([img[:, :3], depth_blurred, img[:, 4:]], dim=1) # [batch_size, c + 1, h, w]

        if self.augment_pipe is not None:
            img = self.augment_pipe(img, num_color_channels=self.G.img_channels) # [batch_size, c, h, w]
        logits, feats = self.D(img, c, update_emas=update_emas, **kwargs)
        return logits, feats

    def extract_patches(self, img: torch.Tensor):
        patch_params = sample_patch_params(len(img), self.patch_cfg, device=img.device)
        img = extract_patches(img, patch_params, resolution=self.patch_cfg.resolution) # [batch_size, c, h_patch, w_patch]
        return img, patch_params

    def compute_sample_weights(self, patch_params, scale_pow: float=1):
        if not self.patch_cfg.enabled:
            return 1.0 # All weigh the same
        """Reweights the distances given the patch params"""
        image_scales = patch_params['scales'].mean(dim=1) # [batch_size]
        sample_weights_raw = image_scales ** scale_pow # [batch_size]
        sample_weights = sample_weights_raw / (sample_weights_raw.mean(dim=0) + 1e-8) # [batch_size]
        return sample_weights # [batch_size]

    def accumulate_gradients(self, phase, real_data: TensorGroup, gen_data: TensorGroup, gain: int, cur_nimg: int):
        assert phase in ['Gmain', 'Greg_pl', 'Gall', 'Dmain', 'Dreg', 'Dall']
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dall': 'Dmain'}.get(phase, phase)

        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        real_data.depth = maybe_blur(real_data.depth, self.cfg.training.blur_real_depth_sigma) # [batch_size, 1, h, w]

        if self.cfg.training.use_depth:
            real_data.img = torch.cat([real_data.img, real_data.depth], dim=1) # [batch_size, c + 1, h, w]

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gall']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_out, patch_params, gen_data.camera_params = self.run_G(gen_data.z, gen_data.c, gen_data.camera_params, camera_angles_cond=gen_data.camera_angles_cond)
                gen_logits, _gen_feats = self.run_D(gen_out.img, gen_data.c, blur_sigma=blur_sigma, patch_params=patch_params, camera_angles=gen_data.camera_params.angles)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                if self.cfg.model.loss_kwargs.adv_loss_type == 'non_saturating':
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                elif self.cfg.model.loss_kwargs.adv_loss_type == 'hinge':
                    loss_Gmain = -gen_logits
                else:
                    raise NotImplementedError(f"Unknown loss: {self.cfg.model.loss_kwargs.adv_loss_type}")
                training_stats.report('Loss/G/loss', loss_Gmain)

            # Lipschitz regularization for camera adaptor
            if self.cfg.training.learn_camera_dist and self.cfg.model.generator.camera_adaptor.lipschitz_weights.enabled:
                with torch.autograd.profiler.record_function('Gmain_camera_forward_for_lipschitz'):
                    # Sampling prior, then unrolling, setting `.requires_grad` and rolling back
                    z = torch.randn(256, self.G.z_dim, device=self.device) # [num_samples, z_dim]
                    c = sample_random_c(len(z), self.G.c_dim, self.device) # [num_samples, c_dim]
                    camera_params_prior = self.G.synthesis.camera_adaptor.sample_from_prior(len(z), device=z.device) # [num_samples, ...]
                    camera_params_prior_raw = self.G.synthesis.camera_adaptor.unroll_camera_params(camera_params_prior) # [num_samples, 8]
                    camera_params_prior_raw.requires_grad_(True) # [num_samples, 8]
                    camera_params_prior = self.G.synthesis.camera_adaptor.roll_camera_params(camera_params_prior_raw) # [num_samples, 8]

                    # Sampling posterior
                    camera_params_posterior = self.G.synthesis.camera_adaptor(camera_params_prior, z, c) # [num_samples, ...]
                    camera_params_posterior_raw = self.G.synthesis.camera_adaptor.unroll_camera_params(camera_params_posterior) # [num_samples, 8]
                    compute_grad = lambda i: torch.autograd.grad(outputs=[camera_params_posterior_raw[:, i].sum()], inputs=[camera_params_prior_raw], create_graph=True, only_inputs=True)[0][:, i]
                    camera_params_grads = torch.stack([compute_grad(i) for i in range(camera_params_posterior_raw.shape[1])], dim=1) # [num_samples, 8]
                    camera_params_grad_norms = camera_params_grads.abs() # [num_samples, 8]
                    lipschitz_regs = (camera_params_grad_norms + 1.0 / (camera_params_grad_norms + 1e-4)).mean(dim=0, keepdim=True) # [1, 8]
                    lipschitz_regs = self.G.synthesis.camera_adaptor.roll_camera_params(lipschitz_regs) # [1, ...]

                    training_stats.report('Dist_lipschitz_reg/yaw', lipschitz_regs.angles[:, 0])
                    training_stats.report('Dist_lipschitz_reg/pitch', lipschitz_regs.angles[:, 1])
                    training_stats.report('Dist_lipschitz_reg/fov', lipschitz_regs.fov)
                    training_stats.report('Dist_lipschitz_reg/radius', lipschitz_regs.radius)
                    training_stats.report('Dist_lipschitz_reg/look_at_yaw', lipschitz_regs.look_at[:, 0])
                    training_stats.report('Dist_lipschitz_reg/look_at_pitch', lipschitz_regs.look_at[:, 1])
                    training_stats.report('Dist_lipschitz_reg/look_at_radius', lipschitz_regs.look_at[:, 2])

                    lipschitz_regs = lipschitz_regs + lipschitz_regs.max() * 0.0
                    lipschitz_regs.angles = lipschitz_regs.angles * self.cfg.model.generator.camera_adaptor.lipschitz_weights.angles # [1, 8]
                    lipschitz_regs.radius = lipschitz_regs.radius * self.cfg.model.generator.camera_adaptor.lipschitz_weights.radius # [1, 8]
                    lipschitz_regs.fov = lipschitz_regs.fov * self.cfg.model.generator.camera_adaptor.lipschitz_weights.fov # [1, 8]
                    lipschitz_regs.look_at = lipschitz_regs.look_at * self.cfg.model.generator.camera_adaptor.lipschitz_weights.look_at # [1, 8]
                    loss_Gcamera_lipschitz = lipschitz_regs.angles[:, :2].sum() + lipschitz_regs.radius.sum() + lipschitz_regs.fov.sum() + lipschitz_regs.look_at.sum() # [1]

                    training_stats.report('Loss/camera_dist/lipschitz_loss', loss_Gcamera_lipschitz)
            else:
                loss_Gcamera_lipschitz = 0.0

            # EMD regularization for camera adaptor
            if self.cfg.training.learn_camera_dist and self.cfg.model.generator.camera_adaptor.emd.enabled and self.emd_multiplier > 0.0:
                with torch.autograd.profiler.record_function('Gmain_camera_forward_for_emd'):
                    # Sampling prior, then unrolling, setting `.requires_grad` and rolling back
                    z = torch.randn(self.cfg.model.generator.camera_adaptor.emd.num_samples, self.G.z_dim, device=self.device) # [num_samples, z_dim]
                    c = sample_random_c(len(z), self.G.c_dim, self.device) # [num_samples, c_dim]
                    camera_params_prior = self.G.synthesis.camera_adaptor.sample_from_prior(len(z), device=z.device) # [num_samples, ...]
                    camera_params_prior_raw = self.G.synthesis.camera_adaptor.unroll_camera_params(camera_params_prior) # [num_samples, 8]
                    camera_params_prior_raw.requires_grad_(True) # [num_samples, 8]
                    camera_params_prior = self.G.synthesis.camera_adaptor.roll_camera_params(camera_params_prior_raw) # [num_samples, 8]

                    # Sampling posterior
                    camera_params_posterior = self.G.synthesis.camera_adaptor(camera_params_prior, z, c) # [num_samples, ...]
                    camera_params_posterior_raw = self.G.synthesis.camera_adaptor.unroll_camera_params(camera_params_posterior) # [num_samples, 8]
                    distance_matrices = torch.stack([ot.dist(camera_params_posterior_raw[:, [i]], camera_params_prior_raw[:, [i]]) for i in range(camera_params_posterior_raw.shape[1])]) # [8, num_samples, num_samples]
                    sample_weights = torch.ones(len(camera_params_posterior_raw), device=self.device) / len(camera_params_posterior_raw) # [num_samples]
                    emd_regs = torch.stack([ot.emd2(sample_weights, sample_weights, M) for M in distance_matrices]) # [8]
                    emd_regs = self.G.synthesis.camera_adaptor.roll_camera_params(emd_regs.unsqueeze(0)) # [1, ...]

                    training_stats.report('Dist_emd_reg/yaw', emd_regs.angles[:, 0])
                    training_stats.report('Dist_emd_reg/pitch', emd_regs.angles[:, 1])
                    training_stats.report('Dist_emd_reg/fov', emd_regs.fov)
                    training_stats.report('Dist_emd_reg/radius', emd_regs.radius)
                    training_stats.report('Dist_emd_reg/look_at_yaw', emd_regs.look_at[:, 0])
                    training_stats.report('Dist_emd_reg/look_at_pitch', emd_regs.look_at[:, 1])
                    training_stats.report('Dist_emd_reg/look_at_radius', emd_regs.look_at[:, 2])

                    emd_regs = emd_regs + emd_regs.max() * 0.0
                    emd_regs.angles = emd_regs.angles * self.cfg.model.generator.camera_adaptor.emd.origin # [1, 3]
                    emd_regs.radius = emd_regs.radius * self.cfg.model.generator.camera_adaptor.emd.radius # [1, 1]
                    emd_regs.fov = emd_regs.fov * self.cfg.model.generator.camera_adaptor.emd.fov # [1, 1]
                    emd_regs.look_at = emd_regs.look_at * self.cfg.model.generator.camera_adaptor.emd.look_at # [1, 3]
                    # Using only the first two angles, radius, fov and look_at.
                    loss_Gcamera_emd = self.emd_multiplier * (emd_regs.angles[:, :2].sum() + emd_regs.radius.sum() + emd_regs.fov.sum() + emd_regs.look_at.sum()) # [1]

                    training_stats.report('Loss/camera_dist/emd_loss', loss_Gcamera_emd)
            else:
                loss_Gcamera_emd = 0.0

            # Simple regularization for the camera adaptor which forces the camera averages towards the mean
            if self.cfg.training.learn_camera_dist and self.cfg.model.generator.camera_adaptor.adjust.angles and self.cfg.model.generator.camera_adaptor.force_mean_weight > 0:
                mean_angles = torch.tensor(get_mean_angles_values(self.cfg.camera.origin.angles)).to(self.device) # [3]
                z = torch.randn(256, self.G.z_dim, device=self.device) # [num_samples, z_dim]
                c = sample_random_c(len(z), self.G.c_dim, self.device) # [num_samples, c_dim]
                camera_params_prior = self.G.synthesis.camera_adaptor.sample_from_prior(len(z), device=self.device) # [num_samples, ...]
                camera_params_posterior = self.G.synthesis.camera_adaptor(camera_params_prior, z, c) # [num_samples, ...]
                loss_Gcamera_force_mean_angles_raw = (camera_params_posterior.angles.mean(dim=0) - mean_angles + 1e-8).square().sum().sqrt() # [1]
                loss_Gcamera_force_mean_angles = self.cfg.model.generator.camera_adaptor.force_mean_weight * loss_Gcamera_force_mean_angles_raw # [1]
                loss_Gcamera_force_mean = loss_Gcamera_force_mean_angles + 0.0 * camera_params_posterior.max() # [1]
                training_stats.report('Loss/camera_dist/force_mean', loss_Gcamera_force_mean_angles)
            else:
                loss_Gcamera_force_mean = 0.0

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Gcamera_lipschitz + loss_Gcamera_force_mean + loss_Gcamera_emd).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg_pl', 'Gall'] and self.cfg.model.loss_kwargs.pl_weight > 0 and (cur_nimg >= self.cfg.model.loss_kwargs.pl_start_kimg * 1000):
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_data.z.shape[0] // self.pl_batch_shrink
                gen_out, _patch_params, gen_data.camera_params = self.run_G(gen_data.z[:batch_size], gen_data.c[:batch_size], gen_data.camera_params[:batch_size], camera_angles_cond=gen_data.camera_angles_cond[:batch_size])
                pl_noise = torch.randn_like(gen_out.img) / np.sqrt(gen_out.img.shape[2] * gen_out.img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_out.img * pl_noise).sum()], inputs=[gen_out.ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.cfg.model.loss_kwargs.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dall']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                with torch.no_grad():
                    gen_out, patch_params, gen_data.camera_params = self.run_G(gen_data.z, gen_data.c, gen_data.camera_params, camera_angles_cond=gen_data.camera_angles_cond, update_emas=True)
                gen_logits, _gen_feats = self.run_D(gen_out.img, gen_data.c, blur_sigma=blur_sigma, update_emas=True, patch_params=patch_params, camera_angles=gen_data.camera_params.angles)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                if self.cfg.model.loss_kwargs.adv_loss_type == 'non_saturating':
                    loss_Dgen = torch.nn.functional.softplus(gen_logits.clamp(min=-self.cfg.model.discriminator.logits_clamp_val, max=None)) # -log(1 - sigmoid(gen_logits))
                    loss_Dgen = loss_Dgen + 0.0 * gen_logits.max() # [batch_size]
                elif self.cfg.model.loss_kwargs.adv_loss_type == 'hinge':
                    loss_Dgen = F.relu(1.0 + gen_logits)
                else:
                    raise NotImplementedError(f"Unknown loss: {self.cfg.model.loss_kwargs.adv_loss_type}")
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dall']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            do_Dkd = self.D_kd_weight > 0 and phase in ['Dmain', 'Dall']
            with torch.autograd.profiler.record_function(name + '_forward'):
                (real_data.img, patch_params) = self.extract_patches(real_data.img) if self.patch_cfg.enabled else (real_data.img, None)
                real_img_tmp = real_data.img.detach().requires_grad_(phase in ['Dreg', 'Dall'])
                real_logits, real_feats = self.run_D(
                    real_img_tmp, real_data.c, blur_sigma=blur_sigma, patch_params=patch_params,
                    camera_angles=real_data.camera_angles, predict_feat=do_Dkd)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                if phase in ['Dmain', 'Dall']:
                    if self.cfg.model.loss_kwargs.adv_loss_type == 'non_saturating':
                        loss_Dreal = torch.nn.functional.softplus(-real_logits.clamp(min=None, max=self.cfg.model.discriminator.logits_clamp_val)) # -log(sigmoid(real_logits))
                        loss_Dreal = loss_Dreal + 0.0 * real_logits.max() # [batch_size]
                    elif self.cfg.model.loss_kwargs.adv_loss_type == 'hinge':
                        loss_Dreal = F.relu(1.0 - real_logits)
                    else:
                        raise NotImplementedError(f"Unknown loss: {self.cfg.model.loss_kwargs.adv_loss_type}")
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                else:
                    loss_Dreal = 0.0

                if do_Dkd:
                    if self.cfg.model.loss_kwargs.kd.discr.loss_type == 'l2':
                        distances = (real_feats - real_data.embs).norm(dim=1) # [batch_size]
                    elif self.cfg.model.loss_kwargs.kd.discr.loss_type == 'kl':
                        distances = torch.nn.functional.kl_div(real_feats.log_softmax(dim=1), real_data.embs.softmax(dim=1), reduction='none').sum(dim=1) # [batch_size]
                    else:
                        raise NotImplementedError(f'Unknown loss type: {self.cfg.model.loss_kwargs.kd.discr.loss_type}')
                    distances = distances * self.compute_sample_weights(patch_params) # [batch_size]
                    loss_Dkd = distances * self.D_kd_weight # [batch_size]
                    training_stats.report('Loss/kd/D_dist', distances)
                    training_stats.report('Loss/kd/D_loss', loss_Dkd)
                else:
                    assert real_feats is None, f"There is no sense in predicting features from D"
                    loss_Dkd = 0.0

                if phase in ['Dreg', 'Dall']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3]) # [batch_size]
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2) # [batch_size]
                    training_stats.report('Loss/D/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
                else:
                    loss_Dr1 = 0.0

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1 + loss_Dkd).mean().mul(gain).backward()

#----------------------------------------------------------------------------

def maybe_blur(img: torch.Tensor, blur_sigma: float) -> torch.Tensor:
    """Blurs the image with a Gaussian filter"""
    blur_size = np.floor(blur_sigma * 3)
    if blur_size > 0:
        f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
        img = upfirdn2d.filter2d(img, f / f.sum())
    return img

#----------------------------------------------------------------------------
