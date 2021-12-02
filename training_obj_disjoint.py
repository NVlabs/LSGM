# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

from util import utils
from torch.cuda.amp import autocast
from training_obj_joint import update_lr, epoch_logging, start_meters, vae_regularization, dae_regularization


def train_vada_disjoint(train_queue, diffusion, dae, dae_optimizer, vae, vae_optimizer, grad_scalar, global_step,
                        warmup_iters, writer, logging, dae_sn_calculator, vae_sn_calculator, args):
    """ This function implements Algorithm 2 from the LSGM paper. It trains both VAE architecture and
    the SGM prior (dae) with two separate batch of t samples. """

    alpha_i = utils.kl_balancer_coeff(num_scales=vae.num_latent_scales,
                                      groups_per_scale=vae.groups_per_scale, fun='square')
    tr_loss_meter, vae_recon_meter, vae_kl_meter, vae_nelbo_meter, kl_per_group_ema = start_meters()

    dae.train()
    vae.train()
    for step, x in enumerate(train_queue):
        # warm-up lr
        update_lr(args, global_step, warmup_iters, dae_optimizer, vae_optimizer)
        x = utils.common_x_operations(x, args.num_x_bits)

        if args.update_q_ema and global_step > 0:
            # switch to EMA parameters
            dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

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

                ##############################################
                ###### Update the VAE encoder/decoder ########
                ##############################################

                kl_T = 0
                noise_q = torch.randn(size=eps.size(), device='cuda')

                # apply diffusion model for samples generated for q (vae)
                t_q, var_t_q, m_t_q, obj_weight_t_q, _, g2_t_q = \
                    diffusion.iw_quantities(args.batch_size, args.time_eps, args.iw_sample_q, args.iw_subvp_like_vp_sde)
                eps_t_q = diffusion.sample_q(eps, noise_q, var_t_q, m_t_q)

                # run the score model
                mixing_component = diffusion.mixing_component(eps_t_q, var_t_q, t_q, enabled=dae.mixed_prediction)
                pred_params_q = dae(eps_t_q, t_q)
                params = utils.get_mixed_prediction(dae.mixed_prediction, pred_params_q, dae.mixing_logit, mixing_component)
                l2_term_q = torch.square(params - noise_q)
                cross_entropy_per_var = obj_weight_t_q * l2_term_q
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
                q_loss = torch.mean(nelbo_loss) + regularization_q   # vae loss

        # backpropagate q_loss for vae and update vae params, if trained
        if args.train_vae:
            grad_scalar.scale(q_loss).backward()
            utils.average_gradients(vae.parameters(), args.distributed)
            if args.grad_clip_max_norm > 0.:  # apply gradient clipping
                grad_scalar.unscale_(vae_optimizer)
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=args.grad_clip_max_norm)
            grad_scalar.step(vae_optimizer)

        if args.update_q_ema and global_step > 0:
            # switch back to original parameters
            dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        ####################################
        ######  Update the SGM prior #######
        ####################################

        # the interface between VAE and DAE is eps.
        eps = eps.detach()

        dae_optimizer.zero_grad()
        with autocast(enabled=args.autocast_train):
            noise_p = torch.randn(size=eps.size(), device='cuda')
            # get diffusion quantities for p sampling scheme (sgm prior)
            t_p, var_t_p, m_t_p, obj_weight_t_p, _, g2_t_p = \
                diffusion.iw_quantities(args.batch_size, args.time_eps, args.iw_sample_p, args.iw_subvp_like_vp_sde)
            eps_t_p = diffusion.sample_q(eps, noise_p, var_t_p, m_t_p)
            # run the score model
            mixing_component = diffusion.mixing_component(eps_t_p, var_t_p, t_p, enabled=dae.mixed_prediction)
            pred_params_p = dae(eps_t_p, t_p)
            params = utils.get_mixed_prediction(dae.mixed_prediction, pred_params_p, dae.mixing_logit, mixing_component)
            l2_term_p = torch.square(params - noise_p)
            p_objective = torch.sum(obj_weight_t_p * l2_term_p, dim=[1, 2, 3])

            regularization_p, dae_norm_loss, dae_bn_loss, dae_wdn_coeff = dae_regularization(args, dae_sn_calculator)

            p_loss = torch.mean(p_objective) + regularization_p

        # update dae parameters
        grad_scalar.scale(p_loss).backward()
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
            writer.add_scalar('train/q_loss', q_loss - regularization_q, global_step)
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