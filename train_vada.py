# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import numpy as np
import os

from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from nvae import NVAE
from diffusion_discretized import DiffusionDiscretized
from diffusion_continuous import make_diffusion
try:
    from apex.optimizers import FusedAdam
except ImportError:
    print("No Apex Available. Using PyTorch's native Adam. Install Apex for faster training.")
    from torch.optim import Adam as FusedAdam
from util.ema import EMA
from util import utils, datasets
from util.sr_utils import SpectralNormCalculator
from evaluate_diffusion import test_dae_fid, generate_samples_vada, elbo_evaluation
from train_vae import infer_active_variables
from training_obj_joint import train_vada_joint
from training_obj_disjoint import train_vada_disjoint


def main(args):
    # common initialization
    logging, writer = utils.common_init(args.global_rank, args.seed, args.save)

    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs

    # load a pretrained NVAE only for VADA.
    load_vae, load_dae = False, False
    if args.vae_checkpoint != '' or args.vada_checkpoint != '':
        assert not (args.vae_checkpoint != '' and args.vada_checkpoint != ''), 'provide only 1 checkpoint'
        checkpoint_path = args.vada_checkpoint if args.vada_checkpoint != '' else args.vae_checkpoint
        logging.info('loading pretrained vae checkpoint:')
        logging.info(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        stored_args = checkpoint['args']
        utils.override_architecture_fields(args, stored_args, logging)
        load_vae = True and not args.discard_vae_weights
        load_dae = args.vada_checkpoint != '' and not args.discard_dae_weights

    arch_instance_vae = utils.get_arch_cells(args.arch_instance, args.use_se)
    logging.info('args = %s', args)

    vae = NVAE(args, arch_instance_vae)
    if load_vae:
        logging.info('loading weights from vae checkpoint')
        vae.load_state_dict(checkpoint['vae_state_dict'])
    vae = vae.cuda()
    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(vae.parameters(), args.distributed)

    vae_optimizer = FusedAdam(vae.parameters(), args.learning_rate_vae, weight_decay=args.weight_decay, eps=1e-3)

    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_vae)

    # enable mixed prediction for vada
    args.mixed_prediction = True
    num_input_channels = vae.latent_structure()[0]
    dae = utils.get_dae_model(args, num_input_channels)
    if load_dae:
        logging.info('loading weights from dae checkpoint')
        dae.load_state_dict(checkpoint['dae_state_dict'])
    dae = dae.cuda()

    # for VESDE, run one epoch over data and get encodings and estimate sigma2_max based on Song's/Ermon's techniques.
    if args.sde_type == 'vesde':
        assert args.sigma2_min == args.sigma2_0, "VESDE was proposed implicitly assuming sigma2_min = sigma2_0!"
        args = utils.set_vesde_sigma_max(args, vae, train_queue, logging, args.distributed)

    diffusion_cont = make_diffusion(args)
    diffusion_disc = DiffusionDiscretized(args, diffusion_cont.var)

    logging.info('DAE: param size = %fM ', utils.count_parameters_in_M(dae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(dae.parameters(), args.distributed)

    dae_optimizer = FusedAdam(dae.parameters(), args.learning_rate_dae, weight_decay=args.weight_decay, eps=1e-4)
    # add EMA functionality to the optimizer
    dae_optimizer = EMA(dae_optimizer, ema_decay=args.ema_decay)

    dae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_dae)

    # create SN calculator
    vae_sn_calculator = SpectralNormCalculator(custom_conv=True)  # NVAE consists of our own custom conv layer classes
    dae_sn_calculator = SpectralNormCalculator(custom_conv=args.custom_conv_dae)   # NCSN++ mode consists of pytorch conv layers
    if args.train_vae:
        vae_sn_calculator.add_conv_layers(vae)
        vae_sn_calculator.add_bn_layers(vae)
    dae_sn_calculator.add_conv_layers(dae)
    dae_sn_calculator.add_bn_layers(dae)

    grad_scalar = GradScaler(2**10)
    bpd_coeff = utils.get_bpd_coeff(args.dataset)

    # if continue training from a checkpoint
    # useful when training is interrupted.
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        dae.load_state_dict(checkpoint['dae_state_dict'])
        # load dae
        dae = dae.cuda()
        dae_optimizer.load_state_dict(checkpoint['dae_optimizer'])
        dae_scheduler.load_state_dict(checkpoint['dae_scheduler'])
        if 'dae_sn_calculator' in checkpoint:   # for backward compatibility
            dae_sn_calculator.load_state_dict(checkpoint['dae_sn_calculator'], torch.device("cuda"))
        # load vae
        vae.load_state_dict(checkpoint['vae_state_dict'])
        vae = vae.cuda()
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
        if 'vae_sn_calculator' in checkpoint:   # for backward compatibility
            vae_sn_calculator.load_state_dict(checkpoint['vae_sn_calculator'], torch.device("cuda"))
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        global_step = checkpoint['global_step']
        best_score_fid = checkpoint['best_score_fid']
        best_score_nll = checkpoint['best_score_nll']
        logging.info('loaded the model at epoch %d.', init_epoch)
    else:
        global_step, epoch, init_epoch, best_score_fid, best_score_nll = 0, 0, 0, 1e10, 1e10

    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            dae_scheduler.step()
            vae_scheduler.step()

        # remove disabled latent variables by setting their mixing component to a small value
        if epoch == 0 and args.mixed_prediction and args.drop_inactive_var:
            logging.info('inferring active latent variables.')
            is_active = infer_active_variables(train_queue, vae, args, max_iter=1000)
            dae.mixing_logit.data[0, torch.logical_not(is_active), 0, 0] = -15
            dae.is_active = is_active.float().view(1, -1, 1, 1)

        # Logging.
        logging.info('epoch %d', epoch)
        if args.disjoint_training:
            # we may use disjoint training for update q with ema
            assert args.iw_sample_p != args.iw_sample_q or args.update_q_ema, \
                'disjoint training is for the case training objective of p and q are not the same unless q is ' \
                'updated with the EMA parameters.'
            assert args.iw_sample_q in ['ll_uniform', 'll_iw']
            assert args.train_vae, 'disjoint training is used when training both VAE and prior.'

            train_obj, global_step = train_vada_disjoint(train_queue, diffusion_cont, dae, dae_optimizer, vae, vae_optimizer,
                                                         grad_scalar, global_step, warmup_iters, writer, logging,
                                                         dae_sn_calculator, vae_sn_calculator, args)
        else:
            assert not args.update_q_ema, 'q can be training with EMA parameters of prior in disjoint training only.'
            train_obj, global_step = train_vada_joint(train_queue, diffusion_cont, dae, dae_optimizer, vae, vae_optimizer,
                                                      grad_scalar, global_step, warmup_iters, writer, logging,
                                                      dae_sn_calculator, vae_sn_calculator, args)

        logging.info('train_loss %f', train_obj)
        writer.add_scalar('train/loss_epoch', train_obj, global_step)

        # generate samples less frequently
        num_evaluations_nll = 20
        num_evaluations_fid = 20
        num_saves = 100           # more frequent saves
        eval_freq_nll = max(args.epochs // num_evaluations_nll, 1)
        eval_freq_fid = max(args.epochs // num_evaluations_fid, 1)
        save_freq = max(args.epochs // num_saves, 1)

        if (epoch + 1) % eval_freq_nll == 0 or (epoch + 1) % eval_freq_fid == 0 or epoch == (args.epochs - 1):
            fast_ode_param = {'ode_eps': args.train_ode_eps, 'ode_solver_tol': args.train_ode_solver_tol}
            dae.eval()
            vae.eval()
            # switch to EMA parameters
            dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            # generate samples
            n = int(np.floor(np.sqrt(min(64, args.batch_size))))  # cannot generate too many samples on big datasets
            num_samples = n ** 2
            samples_disc, _, _, _ = generate_samples_vada(dae, diffusion_disc, vae, num_samples,
                                                             enable_autocast=args.autocast_eval,
                                                             prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0)
            samples_disc = utils.tile_image(samples_disc, n)
            writer.add_image('generated_disc', samples_disc, global_step)
            samples_ode, nfe, _, _ = generate_samples_vada(dae, diffusion_cont, vae, num_samples,
                                                             enable_autocast=args.autocast_eval,
                                                             ode_eps=fast_ode_param['ode_eps'],
                                                             ode_solver_tol=fast_ode_param['ode_solver_tol'],
                                                             ode_sample=True,
                                                             prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0)
            samples_ode = utils.tile_image(samples_ode, n)
            writer.add_image('generated_ode', samples_ode, global_step)
            writer.add_scalar('ode_sampling_nfe/single_batch', nfe, global_step)
            logging.info('sampled new images using ODE framework with {} func. evaluations (ode error tol.: {}, ode eps: {})'
                         .format(nfe, fast_ode_param['ode_solver_tol'], fast_ode_param['ode_eps']))

            if (epoch + 1) % eval_freq_nll == 0 or epoch == (args.epochs - 1):
                # NLL calculation (ODE-based)
                neg_log_p_ode, _, nfe_nll_ode, _, _ = elbo_evaluation(valid_queue, diffusion_cont, dae, args, vae,
                                                                         max_step=10, ode_eval=True, ode_param_dict=fast_ode_param)
                logging.info('valid NLL (ODE-based) {} bpd (with {} func. evaluations)'.format(neg_log_p_ode * bpd_coeff, nfe_nll_ode))
                writer.add_scalar('val/ode_nll_bpd', neg_log_p_ode * bpd_coeff, epoch)
                writer.add_scalar('val/ode_nll_nat', neg_log_p_ode, epoch)
                writer.add_scalar('val/ode_nll_nfe', nfe_nll_ode, epoch)

                if neg_log_p_ode < best_score_nll:
                    best_score_nll = neg_log_p_ode
                    if args.global_rank == 0:
                        # Because we already swapped EMA parameters, we only save the VAE/DAE models here.
                        logging.info('saving the model for NLL.')
                        checkpoint_file_nll = os.path.join(args.save, 'checkpoint_nll.pt')
                        content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                   'best_score_fid': best_score_fid, 'best_score_nll': best_score_nll,
                                   'dae_state_dict': dae.state_dict(), 'vae_state_dict': vae.state_dict()}
                        torch.save(content, checkpoint_file_nll)

            if (epoch + 1) % eval_freq_fid == 0 or epoch == (args.epochs - 1):
                # FID calculation (ODE-based)
                num_fid_samples = 2500  # use small number of samples for FID calculation during training
                val_fid_ema = test_dae_fid(dae, diffusion_cont, writer, logging, args, num_fid_samples, vae,
                                           ode_param_dict=fast_ode_param)
                logging.info('valid FID (ODE-based) {} (ode error tol.: {}, ode eps: {})'.format(
                    val_fid_ema, fast_ode_param['ode_solver_tol'], fast_ode_param['ode_eps']))
                writer.add_scalar('val/ode_fid', val_fid_ema, epoch)

                if val_fid_ema < best_score_fid:
                    best_score_fid = val_fid_ema
                    if args.global_rank == 0:
                        # Because we already swapped EMA parameters, we only save the VAE/DAE models here.
                        logging.info('saving the model for FID.')
                        checkpoint_file_fid = os.path.join(args.save, 'checkpoint_fid.pt')
                        content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                   'best_score_fid': best_score_fid, 'best_score_nll': best_score_nll,
                                   'dae_state_dict': dae.state_dict(), 'vae_state_dict': vae.state_dict()}
                        torch.save(content, checkpoint_file_fid)

            # switch back to original parameters
            dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        if (epoch + 1) % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'grad_scalar': grad_scalar.state_dict(), 'best_score_fid': best_score_fid, 'best_score_nll': best_score_nll,
                           'dae_state_dict': dae.state_dict(), 'dae_optimizer': dae_optimizer.state_dict(),
                           'dae_scheduler': dae_scheduler.state_dict(), 'vae_state_dict': vae.state_dict(),
                           'vae_optimizer': vae_optimizer.state_dict(), 'vae_scheduler': vae_scheduler.state_dict(),
                           'vae_sn_calculator': vae_sn_calculator.state_dict(), 'dae_sn_calculator': dae_sn_calculator.state_dict()}
                torch.save(content, checkpoint_file)

    # skip the final evaluation
    # useful when evaluate_vada.py is used for the evaluation.
    if args.skip_final_eval:
        writer.close()
        return

    # final evaluation (max samples for FID, full validation set of NLL, both ODE-based and with manual discretization)
    eval_ode_param = {'ode_eps': args.eval_ode_eps, 'ode_solver_tol': args.eval_ode_solver_tol}
    dae.eval()
    vae.eval()

    # switch to EMA parameters
    dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    # NLL (ODE-based)
    neg_log_p_ode, _, nfe_nll_ode, _, _ = elbo_evaluation(valid_queue, diffusion_cont, dae, args, vae, max_step=None,
                                                             ode_eval=True, ode_param_dict=eval_ode_param)
    logging.info('final valid NLL (ODE-based) {} bpd (with {} func. evaluations)'.format(neg_log_p_ode * bpd_coeff, nfe_nll_ode))
    writer.add_scalar('final_val/ode_nll_bpd', neg_log_p_ode * bpd_coeff, epoch)
    writer.add_scalar('final_val/ode_nll_nat', neg_log_p_ode, epoch)
    writer.add_scalar('final_val/ode_nll_nfe', nfe_nll_ode, epoch)

    # ELBO (discretized)
    elbo_disc, _, _, _, _ = elbo_evaluation(valid_queue, diffusion_disc, dae, args, vae, max_step=None, ode_eval=False)
    logging.info('final valid ELBO (discretized) bpd %f', elbo_disc * bpd_coeff)
    writer.add_scalar('final_val/disc_elbo_bpd', elbo_disc * bpd_coeff, epoch)
    writer.add_scalar('final_val/disc_elbo_nat', elbo_disc, epoch)

    # FID calculation (using discretized sampling)
    num_fid_samples = 50000
    val_fid_ema_disc = test_dae_fid(dae, diffusion_disc, writer, logging, args, num_fid_samples, vae)
    logging.info('final valid FID (discretized) %f', val_fid_ema_disc)
    writer.add_scalar('final_val/disc_fid', val_fid_ema_disc, epoch)

    # FID calculation (using ODE-based sampling)
    val_fid_ema_ode = test_dae_fid(dae, diffusion_cont, writer, logging, args, num_fid_samples, vae, eval_ode_param)
    logging.info('final valid FID (ODE-based) {} (ode error tol.: {}, ode eps: {})'.format(
        val_fid_ema_ode, eval_ode_param['ode_solver_tol'], eval_ode_param['ode_eps']))
    writer.add_scalar('final_val/ode_fid', val_fid_ema_ode, epoch)

    # switch back to original parameters
    dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nvae-diff/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'celeba_64', 'celeba_256', 'mnist', 'omniglot',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'lsun_church_256'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nvae-diff/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate_vae', type=float, default=1e-4,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min_vae', type=float, default=1e-5,
                        help='min learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay factor')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm_vae', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--grad_clip_max_norm', type=float, default=0.,
                        help='The maximum norm used in gradient norm clipping (0 applies no clipping).')
    # Diffusion
    parser.add_argument('--learning_rate_dae', type=float, default=3e-4,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min_dae', type=float, default=3e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay_norm_dae', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--custom_conv_dae', action='store_true', default=False,
                        help='Set this argument if conv layers in the SGM prior are custom layers from NVAE.')
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--num_scales_dae', type=int, default=4,
                        help='number of spatial scales in denosing model')
    parser.add_argument('--num_cell_per_scale_dae', type=int, default=2,
                        help='number of cells per scale')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='dimension used for time embeddings')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='number of diffusion steps')
    parser.add_argument('--sigma2_0', type=float, default=0.0,
                        help='initial SDE variance at t=0 (sort of represents Normal perturbation of input data)')
    parser.add_argument('--beta_start', type=float, default=0.1,
                        help='initial beta variance value')
    parser.add_argument('--beta_end', type=float, default=20.0,
                        help='final beta variance value')
    parser.add_argument('--vpsde_power', type=int, default=2,
                        help='vpsde power for power-VPSDE')
    parser.add_argument('--sigma2_min', type=float, default=1e-4,
                        help='initial beta variance value')
    parser.add_argument('--sigma2_max', type=float, default=0.99,
                        help='final beta variance value')
    parser.add_argument('--sde_type', type=str, default='geometric_sde',
                        choices=['geometric_sde', 'vpsde', 'sub_vpsde', 'vesde'],
                        help='what kind of sde type to use when training/evaluating in continuous manner.')
    parser.add_argument('--train_ode_eps', type=float, default=1e-2,
                        help='ODE can only be integrated up to some epsilon > 0.')
    parser.add_argument('--train_ode_solver_tol', type=float, default=1e-4,
                        help='ODE solver error tolerance.')
    parser.add_argument('--eval_ode_eps', type=float, default=1e-5,
                        help='ODE can only be integrated up to some epsilon > 0.')
    parser.add_argument('--eval_ode_solver_tol', type=float, default=1e-5,
                        help='ODE solver error tolerance.')
    parser.add_argument('--time_eps', type=float, default=1e-2,
                        help='During training, t is sampled in [time_eps, 1.].')
    parser.add_argument('--denoising_stddevs', type=str, default='beta', choices=['learn', 'beta', 'beta_post'],
                        help='enables learning the conditional VAE decoder distribution standard deviations')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout probability applied to the denosing model')
    parser.add_argument('--fid_dir', type=str, default='/tmp/nvae-diff/fid-stats',
                        help='A dir to store fid related files')
    parser.add_argument('--mixing_logit_init', type=float, default=-3,
                        help='The initial logit for mixing coefficient.')
    parser.add_argument('--embedding_type', type=str, choices=['positional', 'fourier'], default='positional',
                        help='Type of time embedding')
    parser.add_argument('--embedding_scale', type=float, default=1.,
                        # 'fourier':16, 'positional':1000, backward compatible: 1.
                        help='Embedding scale that is used for rescaling time')
    # NCSN++
    parser.add_argument('--dae_arch', type=str, default='unet', choices=['ncsnpp'],
                        help='Switch between different DAE architectures.')
    parser.add_argument('--fir', action='store_true', default=False,
                        help='Enable FIR upsampling/downsampling')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='none', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    # VADA
    parser.add_argument('--vae_checkpoint', type=str, default='',
                        help='Pretrained VAE checkpoint.')
    parser.add_argument('--vada_checkpoint', type=str, default='',
                        help='Pretrained VADA checkpoint.')
    parser.add_argument('--discard_vae_weights', action='store_true', default=False,
                        help='set true to ignore the vae weights from the checkpoint.')
    parser.add_argument('--discard_dae_weights', action='store_true', default=False,
                        help='set true to ignore the dae weights from the checkpoint.')
    parser.add_argument('--train_vae', action='store_true', default=False,
                        help='set true to train the vae model.')
    parser.add_argument('--iw_sample_p', type=str, default='ll_uniform', choices=['ll_uniform', 'll_iw',
                        'drop_all_uniform', 'drop_all_iw', 'drop_sigma2t_iw', 'rescale_iw', 'drop_sigma2t_uniform'],
                        help='Specifies the weighting mechanism used for training p (sgm prior) and whether or not to use importance sampling')
    parser.add_argument('--iw_sample_q', type=str, default='reweight_p_samples', choices=['reweight_p_samples', 'll_uniform', 'll_iw'],
                        help='Specifies the weighting mechanism used for training q (vae) and whether or not to use importance sampling. '
                             'reweight_p_samples indicates reweight the t samples generated for the prior as done in Algorithm 3.')
    parser.add_argument('--iw_subvp_like_vp_sde', action='store_true', default=False,
                        help='Only relevant when using Sub-VPSDE. When true, use VPSDE-based IW distributions.')
    parser.add_argument('--no_autograd_jvp', action='store_true', default=False,
                        help='Set to true to use backward() instead of grad(). '
                             'Suitable for models with gradient checkpointing.')
    parser.add_argument('--apply_sqrt2_res', action='store_true', default=False,
                        help='Enable mixing residual cells with 1/sqrt(2).')
    parser.add_argument('--drop_inactive_var', action='store_true', default=False,
                        help='Drops inactive latent variables.')
    parser.add_argument('--skip_final_eval', action='store_true', default=False,
                        help='set true to skip the final eval.')
    parser.add_argument('--disjoint_training', action='store_true', default=False,
                        help='When p (sgm prior) and q (vae) have different objectives, trains them in two separate forward calls (Algorithm 2).')
    parser.add_argument('--update_q_ema', action='store_true', default=False,
                        help='Enables updating q with EMA parameters of prior.')
    # second stage VADA KL annealing
    parser.add_argument('--cont_kl_anneal', action='store_true', default=False,
                        help='If true, we continue KL annealing using below setup when training LSGM.')
    parser.add_argument('--kl_anneal_portion_vada', type=float, default=0.1,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion_vada', type=float, default=0.0,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff_vada', type=float, default=0.7,
                        help='The constant value used for min KL coeff')
    parser.add_argument('--kl_max_coeff_vada', type=float, default=1.,
                        help='The constant value used for max KL coeff')
    parser.add_argument('--kl_balance_vada', action='store_true', default=False,
                        help='If true, we use KL balancing during VADA KL annealing.')
    # DDP.
    parser.add_argument('--autocast_train', action='store_true', default=True,
                        help='This flag enables FP16 in training.')
    parser.add_argument('--autocast_eval', action='store_true', default=True,
                        help='This flag enables FP16 in evaluation.')
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=utils.init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        utils.init_processes(0, size, main, args)
