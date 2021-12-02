# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

from util import utils
from torch.cuda.amp import autocast


def train_vada_joint(train_queue, diffusion, dae, dae_optimizer, vae, vae_optimizer, grad_scalar, global_step,
                     warmup_iters, writer, logging, dae_sn_calculator, vae_sn_calculator, args):
    """ This function implements Algorithm 1, 2, 3 together from the LSGM paper. If you are trying to understand
    how this function works for the first time, I would suggest checking training_obj_disjoint.py that implements
    Algorithm 3 in a slightly simpler way. """

    alpha_i = utils.kl_balancer_coeff(num_scales=vae.num_latent_scales,
                                      groups_per_scale=vae.groups_per_scale, fun='square')
    tr_loss_meter, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, kl_per_group_ema = start_meters()

    dae.train()
    vae.train()
    for step, x in enumerate(train_queue):
        # warm-up lr
        update_lr(args, global_step, warmup_iters, dae_optimizer, vae_optimizer)
        x = utils.common_x_operations(x, args.num_x_bits)

        dae_optimizer.zero_grad()
        vae_optimizer.zero_grad()
        with autocast(enabled=args.autocast_train):
            # apply vae:
            with torch.set_grad_enabled(args.train_vae):
                logits, all_log_q, all_eps = vae(x)
                eps = vae.concat_eps_per_scale(all_eps)[0]    # prior is applied at the top scale
                remaining_neg_log_p_total, remaining_neg_log_p_per_ver = \
                    utils.cross_entropy_normal(all_eps[vae.num_groups_per_scale:])
                output = vae.decoder_output(logits)
                vae_recon_loss = utils.reconstruction_loss(output, x, crop=vae.crop_output)
                vae_neg_entropy = utils.sum_log_q(all_log_q)

            kl_T = 0

            noise = torch.randn(size=eps.size(), device='cuda')  # note that this noise value is currently shared!

            # get diffusion quantities for p (sgm prior) sampling scheme and reweighting for q (vae)
            t_p, var_t_p, m_t_p, obj_weight_t_p, obj_weight_t_q, g2_t_p = \
                diffusion.iw_quantities(args.batch_size, args.time_eps, args.iw_sample_p, args.iw_subvp_like_vp_sde)
            eps_t_p = diffusion.sample_q(eps, noise, var_t_p, m_t_p)

            # in case we want to train q (vae) with another batch using a different sampling scheme for times t
            if args.iw_sample_q in ['ll_uniform', 'll_iw']:
                t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                    diffusion.iw_quantities(args.batch_size, args.time_eps, args.iw_sample_q, args.iw_subvp_like_vp_sde)
                eps_t_q = diffusion.sample_q(eps, noise, var_t_q, m_t_q)

                eps_t_p = eps_t_p.detach().requires_grad_(True)
                eps_t = torch.cat([eps_t_p, eps_t_q], dim=0)
                var_t = torch.cat([var_t_p, var_t_q], dim=0)
                t = torch.cat([t_p, t_q], dim=0)
                noise = torch.cat([noise, noise], dim=0)
            else:
                eps_t, m_t, var_t, t, g2_t = eps_t_p, m_t_p, var_t_p, t_p, g2_t_p

            # run the score model
            eps_t.requires_grad_(True)
            mixing_component = diffusion.mixing_component(eps_t, var_t, t, enabled=dae.mixed_prediction)
            pred_params = dae(eps_t, t)
            params = utils.get_mixed_prediction(dae.mixed_prediction, pred_params, dae.mixing_logit, mixing_component)
            l2_term = torch.square(params - noise)

            # unpack separate objectives, in case we want to train q (vae) using a different sampling scheme for times t
            if args.iw_sample_q in ['ll_uniform', 'll_iw']:
                l2_term_p, l2_term_q = torch.chunk(l2_term, chunks=2, dim=0)
                p_objective = torch.sum(obj_weight_t_p * l2_term_p, dim=[1, 2, 3])
                cross_entropy_per_var = obj_weight_t_q * l2_term_q
            else:
                p_objective = torch.sum(obj_weight_t_p * l2_term, dim=[1, 2, 3])
                cross_entropy_per_var = obj_weight_t_q * l2_term

            cross_entropy_per_var += diffusion.cross_entropy_const(args.time_eps)
            cross_entropy = torch.sum(cross_entropy_per_var, dim=[1, 2, 3])
            cross_entropy += remaining_neg_log_p_total  # for remaining scales if there is any
            all_neg_log_p = vae.decompose_eps(cross_entropy_per_var)
            all_neg_log_p.extend(remaining_neg_log_p_per_ver)  # add the remaining neg_log_p
            kl_all_list, kl_vals_per_group, kl_diag_list = utils.kl_per_group_vada(all_log_q, all_neg_log_p)

            # kl coefficient
            if args.cont_kl_anneal:
                kl_coeff = utils.kl_coeff(step=global_step,
                                          total_step=args.kl_anneal_portion_vada * args.num_total_iter,
                                          constant_step=args.kl_const_portion_vada * args.num_total_iter,
                                          min_kl_coeff=args.kl_const_coeff_vada,
                                          max_kl_coeff=args.kl_max_coeff_vada)
            else:
                kl_coeff = 1.0

            # nelbo loss with kl balancing
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all_list, kl_coeff, kl_balance=args.kl_balance_vada, alpha_i=alpha_i)
            nelbo_loss = balanced_kl + vae_recon_loss

            # for reporting
            kl = kl_T + vae_neg_entropy + cross_entropy
            # nelbo_loss = kl_coeff * kl + vae_recon_loss

            # compute regularization terms
            regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = vae_regularization(args, vae_sn_calculator)
            regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff = dae_regularization(args, dae_sn_calculator)

            # regularization = regularization_p + regularization_q
            q_loss = torch.mean(nelbo_loss) + regularization_p + regularization_q   # vae loss
            p_loss = torch.mean(p_objective) + regularization_p                     # sgm prior loss

        # backpropagate q_loss for vae and update vae params, if trained
        if args.train_vae:
            grad_scalar.scale(q_loss).backward(retain_graph=utils.different_p_q_objectives(args.iw_sample_p, args.iw_sample_q))
            utils.average_gradients(vae.parameters(), args.distributed)
            if args.grad_clip_max_norm > 0.:  # apply gradient clipping
                grad_scalar.unscale_(vae_optimizer)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=args.grad_clip_max_norm)
            grad_scalar.step(vae_optimizer)

        # if we use different p and q objectives or are not training the vae, discard gradients and backpropagate p_loss
        if utils.different_p_q_objectives(args.iw_sample_p, args.iw_sample_q) or not args.train_vae:
            if args.train_vae:
                # discard current gradients computed by weighted loss for VAE
                dae_optimizer.zero_grad()

            # compute gradients with unweighted loss
            grad_scalar.scale(p_loss).backward()

        # update dae parameters
        utils.average_gradients(dae.parameters(), args.distributed)
        if args.grad_clip_max_norm > 0.:         # apply gradient clipping
            grad_scalar.unscale_(dae_optimizer)
            torch.nn.utils.clip_grad_norm_(dae.parameters(), max_norm=args.grad_clip_max_norm)
        grad_scalar.step(dae_optimizer)

        # update grade scalar
        grad_scalar.update()

        # Bookkeeping!
        # update average meters
        tr_loss_meter.update(q_loss.data, 1)
        vae_recon_meter.update(torch.mean(vae_recon_loss.data), 1)
        vae_kl_meter.update(torch.mean(kl).data, 1)
        vae_nelbo_meter.update(torch.mean(kl + vae_recon_loss).data, 1)
        kl_per_group_ema.update(kl_vals_per_group.data, 1)

        if (global_step + 1) % 200 == 0:
            writer.add_scalar('train/lr_dae', dae_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/lr_vae', vae_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/q_loss', q_loss - regularization_p - regularization_q, global_step)
            writer.add_scalar('train/p_loss', p_loss - regularization_p, global_step)
            writer.add_scalar('train/norm_loss_vae', vae_norm_loss, global_step)
            writer.add_scalar('train/norm_loss_dae', dae_norm_loss, global_step)
            writer.add_scalar('train/bn_loss_vae', vae_bn_loss, global_step)
            writer.add_scalar('train/bn_loss_dae', dae_bn_loss, global_step)
            writer.add_scalar('train/kl_coeff', kl_coeff, global_step)
            writer.add_scalar('train/norm_coeff_vae', vae_wdn_coeff, global_step)
            writer.add_scalar('train/norm_coeff_dae', dae_wdn_coeff, global_step)
            if (global_step + 1) % 2000 == 0:  # reduced frequency
                if dae.mixed_prediction:
                    m = torch.sigmoid(dae.mixing_logit)
                    if not torch.isnan(m).any():
                        writer.add_histogram('mixing_prob', m, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag_list):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl_active_step/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff_step/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals_step/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl_active_step/total_active', total_active, global_step)
        global_step += 1

    # write at the end of epoch
    epoch_logging(args, writer, global_step, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, tr_loss_meter, kl_per_group_ema)

    utils.average_tensor(tr_loss_meter.avg, args.distributed)
    return tr_loss_meter.avg, global_step


def vae_regularization(args, vae_sn_calculator):
    regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff = 0., 0., 0., args.weight_decay_norm_vae
    if args.train_vae:
        vae_norm_loss = vae_sn_calculator.spectral_norm_parallel()
        vae_bn_loss = vae_sn_calculator.batchnorm_loss()
        regularization_q = (vae_norm_loss + vae_bn_loss) * vae_wdn_coeff

    return regularization_q, vae_norm_loss, vae_bn_loss, vae_wdn_coeff


def dae_regularization(args, dae_sn_calculator):
    dae_wdn_coeff = args.weight_decay_norm_dae
    dae_norm_loss = dae_sn_calculator.spectral_norm_parallel()
    dae_bn_loss = dae_sn_calculator.batchnorm_loss()
    regularization_p = (dae_norm_loss + dae_bn_loss) * dae_wdn_coeff

    return regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff


def update_lr(args, global_step, warmup_iters, dae_optimizer, vae_optimizer):
    if global_step < warmup_iters:
        lr = args.learning_rate_dae * float(global_step) / warmup_iters
        for param_group in dae_optimizer.param_groups:
            param_group['lr'] = lr

        if args.train_vae:
            lr = args.learning_rate_vae * float(global_step) / warmup_iters
            for param_group in vae_optimizer.param_groups:
                param_group['lr'] = lr


def start_meters():
    tr_loss_meter = utils.AvgrageMeter()
    vae_recon_meter = utils.AvgrageMeter()
    vae_kl_meter = utils.AvgrageMeter()
    vae_nelbo_meter = utils.AvgrageMeter()
    kl_per_group_ema = utils.AvgrageMeter()
    return tr_loss_meter, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, kl_per_group_ema


def epoch_logging(args, writer, step, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, tr_loss_meter, kl_per_group_ema):
    utils.average_tensor(vae_recon_meter.avg, args.distributed)
    utils.average_tensor(vae_kl_meter.avg, args.distributed)
    utils.average_tensor(vae_nelbo_meter.avg, args.distributed)
    utils.average_tensor(tr_loss_meter.avg, args.distributed)
    utils.average_tensor(kl_per_group_ema.avg, args.distributed)

    writer.add_scalar('epoch/vae_recon', vae_recon_meter.avg, step)
    writer.add_scalar('epoch/vae_kl', vae_kl_meter.avg, step)
    writer.add_scalar('epoch/vae_nelbo', vae_nelbo_meter.avg, step)
    writer.add_scalar('epoch/total_loss', tr_loss_meter.avg, step)
    # add kl value per group to tensorboard
    for i in range(len(kl_per_group_ema.avg)):
        writer.add_scalar('kl_value/group_%d' % i, kl_per_group_ema.avg[i], step)