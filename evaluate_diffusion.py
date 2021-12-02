# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import torch
from timeit import default_timer as timer

from util import utils
from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3
from torch.cuda.amp import autocast
from diffusion_continuous import DiffusionBase
from diffusion_discretized import DiffusionDiscretized


def create_generator_vada(dae, diffusion, vae, batch_size, num_total_samples, enable_autocast, ode_param_dict, prior_var, temp, vae_temp):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    shape = [dae.num_input_channels, dae.input_size, dae.input_size]
    for i in range(num_iters):
        with torch.no_grad():
            if ode_param_dict is None:
                eps = diffusion.run_denoising_diffusion(dae, batch_size, shape, temp, enable_autocast, is_image=False, prior_var=prior_var)
            else:
                eps, _, _ = diffusion.sample_model_ode(dae, batch_size, shape, ode_param_dict['ode_eps'],
                                                         ode_param_dict['ode_solver_tol'], enable_autocast, temp)
            decomposed_eps = vae.decompose_eps(eps)
            image = vae.sample(batch_size, vae_temp, decomposed_eps, enable_autocast)
        yield image.float()


def generate_samples_vada(dae, diffusion, vae, num_samples, enable_autocast, ode_eps=None, ode_solver_tol=None,
                          ode_sample=False, prior_var=1.0, temp=1.0, vae_temp=1.0, noise=None):
    shape = [dae.num_input_channels, dae.input_size, dae.input_size]
    with torch.no_grad():
        if ode_sample:
            assert isinstance(diffusion, DiffusionBase), 'ODE-based sampling requires cont. diffusion!'
            assert ode_eps is not None, 'ODE-based sampling requires integration cutoff ode_eps!'
            assert ode_solver_tol is not None, 'ODE-based sampling requires ode solver tolerance!'
            start = timer()
            eps, nfe, time_ode_solve = diffusion.sample_model_ode(dae, num_samples, shape, ode_eps, ode_solver_tol, enable_autocast, temp, noise)
        else:
            assert isinstance(diffusion, DiffusionDiscretized), 'Regular sampling requires disc. diffusion!'
            assert noise is None, 'Noise is not used in ancestral sampling.'
            nfe = diffusion._diffusion_steps
            time_ode_solve = 999.999  # Yeah I know...
            start = timer()
            eps = diffusion.run_denoising_diffusion(dae, num_samples, shape, temp, enable_autocast, is_image=False, prior_var=prior_var)
        decomposed_eps = vae.decompose_eps(eps)
        image = vae.sample(num_samples, vae_temp, decomposed_eps, enable_autocast)
        end = timer()
        sampling_time = end - start
        # average over GPUs
        nfe_torch = torch.tensor(nfe * 1.0, device='cuda')
        sampling_time_torch = torch.tensor(sampling_time * 1.0, device='cuda')
        time_ode_solve_torch = torch.tensor(time_ode_solve * 1.0, device='cuda')
        utils.average_tensor(nfe_torch, True)
        utils.average_tensor(sampling_time_torch, True)
        utils.average_tensor(time_ode_solve_torch, True)
    return image, nfe_torch, time_ode_solve_torch, sampling_time_torch


def elbo_evaluation(val_queue, diffusion, dae, args, vae=None, max_step=None, ode_eval=False, ode_param_dict=None,
                    num_samples=1, num_inner_samples=1, report_std=False):
    nelbo_avg, neg_log_p_avg = utils.AvgrageMeter(), utils.AvgrageMeter()
    if ode_eval:
        # Note that we are currently not averaging the NFE counter over different GPUs! Doesn't seem very important,
        # though, as NFEs mainly matter  for sampling not NLL calculation.
        nfe_counter_avg = utils.AvgrageMeter()

    if ode_eval and num_inner_samples > 1 and report_std:
        stddev_avg = utils.AvgrageMeter()
        stderror_avg = utils.AvgrageMeter()

    dae.eval()
    vae.eval()
    with torch.no_grad():
        for step, x in enumerate(val_queue):
            # we avoid computing ELBO on the whole dataset
            if max_step is not None and step >= max_step:
                break

            x = utils.common_x_operations(x, args.num_x_bits)

            nelbo, log_iw = [], []
            for k in range(num_samples):
                with autocast(enabled=args.autocast_eval):
                    # apply vae:
                    logits, all_log_q, all_eps = vae(x)
                    eps = vae.concat_eps_per_scale(all_eps)[0]  # prior is applied at the top scale
                    remaining_neg_log_p_total, _ = utils.cross_entropy_normal(all_eps[vae.num_groups_per_scale:])
                    output = vae.decoder_output(logits)
                    vae_recon_loss = utils.reconstruction_loss(output, x, crop=vae.crop_output)
                    neg_vae_entropy = utils.sum_log_q(all_log_q)
                    vae_loss = vae_recon_loss + neg_vae_entropy + remaining_neg_log_p_total

                # computing prior likelihood outside of autocast as the inner functions have their own autocast
                if ode_eval:
                    assert isinstance(diffusion, DiffusionBase), 'ODE-based NLL evaluation requires cont. diffusion!'
                    assert ode_param_dict is not None
                    nelbo_prior, nfe, stddev_batch, stderror_batch = diffusion.compute_ode_nll(
                        dae=dae, eps=eps, ode_eps=ode_param_dict['ode_eps'],
                        ode_solver_tol=ode_param_dict['ode_solver_tol'], enable_autocast=args.autocast_eval,
                        no_autograd=args.no_autograd_jvp, num_samples=num_inner_samples, report_std=report_std)

                    nfe_counter_avg.update(nfe, x.size(0))
                    if num_inner_samples > 1 and report_std:
                        assert stddev_batch is not None and stderror_batch is not None
                        stddev_avg.update(stddev_batch, x.size(0))
                        stderror_avg.update(stderror_batch, x.size(0))
                else:
                    assert isinstance(diffusion, DiffusionDiscretized), 'Regular NLL evaluation requires disc. diffusion!'
                    assert num_inner_samples == 1, 'inner_samples more than one is not implemented'
                    nelbo_prior = diffusion.compute_nelbo(dae, eps, enable_autocast=args.autocast_eval, is_image=False,
                                                          prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0)

                nelbo_k = nelbo_prior + vae_loss
                nelbo.append(nelbo_k)
                log_iw.append(-nelbo_k)   # we can use nelbo as the KL is computed using a sample based objective.

            # IW estimation of log prob
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))
            loss_iw = torch.mean(-log_p)
            neg_log_p_avg.update(loss_iw.data, x.size(0))

            # Multi-sample estimation of nelbo
            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            loss_nelbo = torch.mean(nelbo)
            nelbo_avg.update(loss_nelbo.data, x.size(0))

    if ode_eval and num_inner_samples > 1 and report_std:
        utils.average_tensor(stddev_avg.avg, args.distributed)
        utils.average_tensor(stderror_avg.avg, args.distributed)
        stddev_avg_return = stddev_avg.avg
        stderror_avg_return = stderror_avg.avg
    else:
        stddev_avg_return = None
        stderror_avg_return = None

    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    utils.average_tensor(nelbo_avg.avg, args.distributed)
    nfes = nfe_counter_avg.avg if ode_eval else None
    return nelbo_avg.avg, neg_log_p_avg.avg, nfes, stddev_avg_return, stderror_avg_return


def test_dae_fid(model, diffusion, writer, logging, args, total_fid_samples, vae=None, ode_param_dict=None,
                 temp=1.0, vae_temp=1.0):
    dims = 2048
    device = 'cuda'
    enable_autocast = args.autocast_eval
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))
    if vae is not None:
        prior_var = args.sigma2_max if args.sde_type == 'vesde' else 1.0
        g = create_generator_vada(model, diffusion, vae, args.batch_size, num_sample_per_gpu, enable_autocast, ode_param_dict,
                                  prior_var, temp, vae_temp)
    else:
        raise NotImplementedError('vae cannot be None.')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid
