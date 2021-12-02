# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import torch
from torch.cuda.amp import autocast
import numpy as np
from util import utils


class DiffusionDiscretized(object):
    """
    This class constructs the diffusion process and provides all related methods and constants.
    """
    def __init__(self, args, var_fun):  # alpha_bars_fun
        self._diffusion_steps = args.diffusion_steps
        self._denoising_stddevs = args.denoising_stddevs
        self._var_fun = var_fun

        self._betas_init, self._alphas, self._alpha_bars, self._betas_post_init = \
            self._generate_base_constants(diffusion_steps=self._diffusion_steps)

        self._weights_invsigma2_term, self._weights_const_term, self._weights_noisepred_l2_term = \
            self._generate_reconstruction_weights()

    def _generate_base_constants(self, diffusion_steps):
        """
        Generates torch tensors with basic constants for all timesteps.
        """
        betas_np = self._generate_betas_from_continuous_fun(diffusion_steps)

        alphas_np = 1.0 - betas_np
        alpha_bars_np = np.cumprod(alphas_np)

        # posterior variances only make sense for t>1, hence the array is short by 1
        betas_post_np = betas_np[1:] * (1.0 - alpha_bars_np[:-1]) / (1.0 - alpha_bars_np[1:])
        # we add beta_post_2 to the beginning of both beta arrays, since this is used as final decoder variance and
        # requires special treatment (as in diffusion paper)
        betas_post_init_np = np.append(betas_post_np[0], betas_post_np)

        betas_init = torch.from_numpy(betas_np).float().cuda()
        alphas = torch.from_numpy(alphas_np).float().cuda()
        alpha_bars = torch.from_numpy(alpha_bars_np).float().cuda()
        betas_post_init = torch.from_numpy(betas_post_init_np).float().cuda()

        return betas_init, alphas, alpha_bars, betas_post_init

    def _generate_betas_from_continuous_fun(self, diffusion_steps):
        t = np.arange(1, diffusion_steps + 1, dtype=np.float64)
        t = t / diffusion_steps

        # alpha_bars = self._alpha_bars_fun(t)
        alpha_bars = 1.0 - self._var_fun(torch.tensor(t)).numpy()
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = np.hstack((1 - alpha_bars[0], betas))

        return betas

    def _generate_reconstruction_weights(self):
        """
        Generates torch tensors with weights for the different terms in the KL and log_prob reconstruction objectives
        for all timesteps.
        """
        weights_invsigma2_term = torch.cat((torch.tensor([0.0], device='cuda'), self._betas_post_init[1:]))

        const_term0 = torch.tensor([0.5 * np.log(2.0 * np.pi)], device='cuda')
        const_terms = -0.5 * torch.log(self._betas_post_init[1:]) - 0.5
        weights_const_term = torch.cat((const_term0, const_terms))

        noisepred_l2_terms = torch.square(self._betas_init) / (self._alphas * (1.0 - self._alpha_bars))
        weights_noisepred_l2_term = noisepred_l2_terms

        return weights_invsigma2_term, weights_const_term, weights_noisepred_l2_term

    def get_fixed_base_log_scale(self):
        """
        Get log std dev. for learning base distribution with fixed variance.
        """
        return 0.5 * torch.log(1.0 - self._alpha_bars[-1])

    def get_p_log_scales(self, timestep, stddev_type):
        """
        Grab log std devs. of backward denoising process p, if we decide to fix them.
        """
        if stddev_type == 'beta':
            # use diffusion variances, except for t=1, for which we use posterior variance beta_post_2
            return 0.5 * torch.log(torch.gather(self._betas_init, 0, timestep-1))
        elif stddev_type == 'beta_post':
            # use diffusion posterior variances, except for t=1, for which there is no posterior, so we use beta_post_2
            return 0.5 * torch.log(torch.gather(self._betas_post_init, 0, timestep-1))
        elif stddev_type == 'learn':
            return None
        else:
            raise ValueError('Unknown stddev_type: {}'.format(stddev_type))

    def sample_q(self, x_init, noise, timestep):
        """
        Sample from diffusion process given inital images batch, independent noise samples, and target timesteps.
        """
        assert x_init.shape[0] == timestep.shape[0] == noise.shape[0]
        size = x_init.size()
        alpha_bars = torch.gather(self._alpha_bars, 0, timestep-1)
        alpha_bars_sqrt = utils.view4D(torch.sqrt(alpha_bars), size)
        one_minus_alpha_bars_sqrt = utils.view4D(torch.sqrt(1.0 - alpha_bars), size)
        return alpha_bars_sqrt * x_init + one_minus_alpha_bars_sqrt * noise

    def _extract_loss_weights(self, timestep, size):
        """
        Auxiliary function to calculate individual loss term weights using model's alpha and beta constants.
        """
        assert timestep is not None
        weights_invsigma2_term = utils.view4D(torch.gather(self._weights_invsigma2_term, 0, timestep-1), size)
        weights_const_term = utils.view4D(torch.gather(self._weights_const_term, 0, timestep-1), size)
        weights_l2_term = utils.view4D(torch.gather(self._weights_noisepred_l2_term, 0, timestep-1), size)

        return weights_l2_term, weights_invsigma2_term, weights_const_term

    def reconstruction_loss(self, decoder, x, timestep=None, weighted_loss=False):
        """
        Our reconstruction losses are not just regular log probabilities, but come from closed form KLs for t>1.
        This requires special treatment and hence the diffusion model provides its own reconstruction loss routine.
        """
        means, log_stds, num_c = decoder.get_params()
        l2_term = torch.square(means - x)

        if weighted_loss:
            size = l2_term.size()
            weights_l2_term, weights_invsigma2_term, weights_const_term = self._extract_loss_weights(timestep, size)
            assert log_stds.size() == weights_l2_term.size() == weights_invsigma2_term.size() == weights_const_term.size()

            invsigma2_term = 0.5 * torch.exp(-2.0 * log_stds)
            recloss = weights_l2_term * l2_term * invsigma2_term + \
                      weights_invsigma2_term * invsigma2_term + \
                      log_stds + weights_const_term
        else:
            recloss = l2_term

        return torch.sum(recloss, dim=[1, 2, 3])

    def run_denoising_diffusion(self, model, num_samples, shape, temp=1.0, enable_autocast=False, is_image=True, prior_var=1.0):
        """
        Run the full denoising sampling loop.
        """
        # set model to eval mode
        model.eval()

        # initialize sample
        x_noisy_size = [num_samples] + shape
        x_noisy = torch.randn(size=x_noisy_size, device='cuda') * np.sqrt(prior_var) * temp

        # denoising loop
        with torch.no_grad():
            for t in reversed(range(0, self._diffusion_steps)):
                timestep = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)  # the model uses (1 ... T) without 0
                fixed_log_scales = self.get_p_log_scales(timestep=timestep, stddev_type=self._denoising_stddevs)
                mixing_component = self.get_mixing_component(x_noisy, timestep, enabled=model.mixed_prediction)

                # run model
                with autocast(enable_autocast):
                    pred_logits = model(x_noisy, timestep.float() / self._diffusion_steps)
                    logits = utils.get_mixed_prediction(model.mixed_prediction, pred_logits, model.mixing_logit, mixing_component)

                output_dist = utils.decoder_output(model.dataset, logits, fixed_log_scales=fixed_log_scales)
                noise = torch.randn(size=x_noisy_size, device='cuda')
                mean = self.get_q_posterior_mean(x_noisy, output_dist.means, t)
                if t == 0:
                    x_image = mean
                else:
                    x_noisy = mean + torch.exp(output_dist.log_scales) * noise * temp

            if is_image:
                x_image = x_image.clamp(min=-1., max=1.)
                x_image = utils.unsymmetrize_image_data(x_image)
            else:
                x_image = x_image.clamp(min=-3., max=3.)
        return x_image

    def compute_nelbo(self, model, x_init, enable_autocast=False, is_image=True, prior_var=1.0):
        # TODO: reconstruction loss for now computes only the continuous reconstruction in L0:
        assert not is_image

        with torch.no_grad():
            nelbo = self.calculate_base_kl(x_init, prior_var)  # KL0
            for t in reversed(range(0, self._diffusion_steps)):
                num_samples = x_init.size(0)
                timestep = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t + 1)  # the model uses (1 ... T) without 0
                fixed_log_scales = self.get_p_log_scales(timestep=timestep, stddev_type=self._denoising_stddevs)

                # get x_t for diffusion model
                noise = torch.randn(size=x_init.size(), device='cuda')
                x_noisy = self.sample_q(x_init, noise=noise, timestep=timestep)  # conditioning state x_t
                mixing_component = self.get_mixing_component(x_noisy, timestep, enabled=model.mixed_prediction)

                # run model
                with autocast(enable_autocast):
                    pred_logits = model(x_noisy.float(), timestep.float() / self._diffusion_steps)
                    logits = utils.get_mixed_prediction(model.mixed_prediction, pred_logits, model.mixing_logit, mixing_component)
                    output_dist = utils.decoder_output(model.dataset, logits, fixed_log_scales=fixed_log_scales)

                target = noise
                nelbo = nelbo + self.reconstruction_loss(output_dist, target, timestep, weighted_loss=True)

        return nelbo

    def get_q_posterior_mean(self, x_noisy, prediction, t):
        # last step works differently (for better FIDs we NEVER sample in last conditional images output!)
        if t == 0:
            mean = 1.0 / torch.sqrt(self._alpha_bars[0]) * \
                   (x_noisy - torch.sqrt(1.0 - self._alpha_bars[0]) * prediction)
        else:
            mean = 1.0 / torch.sqrt(self._alphas[t]) * \
                      (x_noisy - self._betas_init[t] * prediction /
                       torch.sqrt(1.0 - self._alpha_bars[t]))

        return mean

    def calculate_base_kl(self, x_init, prior_var):
        """
        Calculates the base KL term assuming a Normal N(0, prior_var) p_T base distribution.
        """
        kl = 0.5 * (self._alpha_bars[-1] * torch.square(x_init)
                    + torch.exp(2.0 * self.get_fixed_base_log_scale())) / prior_var \
            - self.get_fixed_base_log_scale() \
            + 0.5 * np.log(prior_var) \
            - 0.5 
        return torch.sum(kl, dim=[1, 2, 3])

    def get_mixing_component(self, x_noisy, timestep, enabled):
        size = x_noisy.size()
        alpha_bars = torch.gather(self._alpha_bars, 0, timestep-1)
        if enabled:
            one_minus_alpha_bars_sqrt = utils.view4D(torch.sqrt(1.0 - alpha_bars), size)
            mixing_component = one_minus_alpha_bars_sqrt * x_noisy
        else:
            mixing_component = None

        return mixing_component


