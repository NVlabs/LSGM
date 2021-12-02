# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
from torchvision.utils import save_image
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time

from torch.multiprocessing import Process

from nvae import NVAE
from diffusion_discretized import DiffusionDiscretized
from diffusion_continuous import make_diffusion
try:
    from apex.optimizers import FusedAdam
except ImportError:
    print("No Apex Available. Using PyTorch's native Adam. Install Apex for faster training.")
    from torch.optim import Adam as FusedAdam
from util import utils, datasets
from util.ema import EMA
from evaluate_diffusion import test_dae_fid, generate_samples_vada, elbo_evaluation


def main(eval_args):
    # common initialization
    logging, writer = utils.common_init(eval_args.global_rank, eval_args.seed, eval_args.save)

    # load a checkpoint
    logging.info('#' * 80)
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']


    # adding some arguments for backward compatibility.
    if not hasattr(args, 'num_x_bits'):
        logging.info('*** Setting %s manually ****', 'num_x_bits')
        setattr(args, 'num_x_bits', 8)

    if not hasattr(args, 'channel_mult'):
        logging.info('*** Setting %s manually ****', 'channel_mult')
        setattr(args, 'channel_mult', [1, 2])

    if not hasattr(args, 'mixing_logit_init'):
        logging.info('*** Setting %s manually ****', 'mixing_logit_init')
        setattr(args, 'mixing_logit_init', -3.0)

    if eval_args.diffusion_steps > 0:
        args.diffusion_steps = eval_args.diffusion_steps

    epoch = checkpoint['epoch']
    logging.info('loaded the model at epoch %d', checkpoint['epoch'])

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)
    logging.info('args = %s', args)
    logging.info('evalargs = %s', eval_args)

    # load VAE
    vae = NVAE(args, arch_instance_nvae)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae = vae.cuda()
    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))

    # load DAE
    num_input_channels = vae.latent_structure()[0]
    dae = utils.get_dae_model(args, num_input_channels)
    dae.load_state_dict(checkpoint['dae_state_dict'])
    diffusion_cont = make_diffusion(args)
    diffusion_disc = DiffusionDiscretized(args, diffusion_cont.var)

    logging.info('DAE: param size = %fM ', utils.count_parameters_in_M(dae))
    checkpoint_name = os.path.basename(eval_args.checkpoint)
    if checkpoint_name == 'checkpoint.pt':
        logging.info('Swapping the parameters of DAE with EMA parameters')
        # checkpoint.pt models require swapping EMA parameters
        dae_optimizer = FusedAdam(dae.parameters(), args.learning_rate_dae,
                                  weight_decay=args.weight_decay, eps=1e-4)
        # add EMA functionality to the optimizer
        dae_optimizer = EMA(dae_optimizer, ema_decay=args.ema_decay)
        dae_optimizer.load_state_dict(checkpoint['dae_optimizer'])

        # replace DAE parameters with EMA values
        dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)
    elif checkpoint_name in {'checkpoint_fid.pt', 'checkpoint_nll.pt', 'checkpoint_finetuned.pt'}:
        logging.info('swapping the parameters of DAE with EMA parameters is ** not ** required.')
    else:
        raise ValueError('Cannot recognize checkpoint name %s' % checkpoint_name)
    dae = dae.cuda()

    # set the model to eval() model.
    dae.eval()
    # set vae to train mode if the arg says
    vae.train(mode=eval_args.vae_train_mode)

    if eval_args.eval_mode == 'evaluate':
        # replace a few fields in args based on eval_args
        # this will allow train/evaluate on different systems
        args.num_proc_node = eval_args.num_proc_node
        args.num_process_per_node = eval_args.num_process_per_node
        args.data = eval_args.data
        ode_param_dict = {'ode_solver_tol': eval_args.ode_solver_tol, 'ode_eps': eval_args.ode_eps}
        if eval_args.batch_size > 0:
            args.batch_size = eval_args.batch_size

        # load train valid queue
        bpd_coeff = utils.get_bpd_coeff(args.dataset)
        train_queue, valid_queue, num_classes = datasets.get_loaders(args)

        if eval_args.eval_on_train:
            logging.info('Using the training data for eval.')
            valid_queue = train_queue

        # evaluate NLL ODE-style
        if eval_args.nll_ode_eval:
            args.ode_eps = eval_args.ode_eps
            args.ode_solver_tol = eval_args.ode_solver_tol
            logging.info('Running ODE-based NLL evaluation...')
            nelbo_ode, neg_log_p_ode, nfe_nll_ode, stddev, stderror = elbo_evaluation(
                valid_queue, diffusion_cont, dae, args, vae, max_step=None, ode_eval=True, ode_param_dict=ode_param_dict,
                num_samples=eval_args.num_iw_samples, num_inner_samples=eval_args.num_iw_inner_samples, report_std=True)
            logging.info('valid ODE-based NELBO: {} bpd (with {} func. evals on average, ode error tol. {}, ode eps {})'
                         .format(nelbo_ode * bpd_coeff, nfe_nll_ode, ode_param_dict['ode_solver_tol'], ode_param_dict['ode_eps']))
            logging.info('valid ODE-based neg. log p: {} bpd'.format(neg_log_p_ode * bpd_coeff))
            if stddev is not None and stderror is not None:
                logging.info('valid ODE-based prior prob std. deviation: {} bpd ({} nats)'.format(stddev * bpd_coeff, stddev))
                logging.info('valid ODE-based prior prob std. error: {} bpd ({} nats)'.format(stderror * bpd_coeff, stderror))
                logging.info('valid ODE-based prior prob variance (based on std. deviation): {} bpd ({} nats)'.format(stddev * stddev * bpd_coeff, stddev * stddev))
                logging.info('valid ODE-based prior prob variance (based on std. error): {} bpd ({} nats)'.format(stderror * stderror * bpd_coeff, stderror * stderror))
            writer.add_scalar('val/final_ode_nelbo_bpd', nelbo_ode * bpd_coeff, 1)
            writer.add_scalar('val/final_ode_neg_log_p_bpd', neg_log_p_ode * bpd_coeff, 1)
            writer.add_scalar('val/final_ode_nfe', nfe_nll_ode, 1)

        # evaluate ELBO with regular fixed discretization
        if eval_args.elbo_eval:
            logging.info('Running regular/discretized ELBO evaluation...')
            val_elbo, val_log_p, _, _, _ = elbo_evaluation(valid_queue, diffusion_disc, dae, args, vae, max_step=None,
                                                              ode_eval=False, num_samples=eval_args.num_iw_samples)
            logging.info('valid NELBO: {} bpd'.format(val_elbo * bpd_coeff))
            logging.info('valid neg log P {} bpd %f'.format(val_log_p * bpd_coeff))
            writer.add_scalar('val/final_nelbo_bpd', val_elbo * bpd_coeff, 1)
            writer.add_scalar('val/final_neg_log_p_bpd', val_log_p * bpd_coeff, 1)

        # compute FID using samples from ODE-based sampling
        if eval_args.fid_ode_eval:
            args.fid_dir = eval_args.fid_dir
            num_fid_samples = eval_args.num_fid_samples
            logging.info('Running ODE-based FID evaluation...')
            val_fid_ema_ode = test_dae_fid(dae, diffusion_cont, writer, logging, args, num_fid_samples, vae,
                                           ode_param_dict, eval_args.temp, eval_args.vae_temp)
            logging.info('valid ODE-based FID: {} (ode error tol. {}, ode eps {})'
                         .format(val_fid_ema_ode, ode_param_dict['ode_solver_tol'], ode_param_dict['ode_eps']))
            writer.add_scalar('val/final_fid_ode_samples', val_fid_ema_ode, 1)

        # compute FID using samples from regular fixed discretization sampling
        if eval_args.fid_disc_eval:
            args.fid_dir = eval_args.fid_dir
            num_fid_samples = eval_args.num_fid_samples
            logging.info('Running regular/discretized FID evaluation...')
            val_fid_ema = test_dae_fid(dae, diffusion_disc, writer, logging, args, num_fid_samples, vae,
                                       temp=eval_args.temp, vae_temp=eval_args.vae_temp)
            logging.info('valid FID: {}'.format(val_fid_ema))
            writer.add_scalar('val/final_fid', val_fid_ema, 1)

        # compute average number of function evaluation for ODE-based sampling from the model
        if eval_args.nfe_eval:
            nfe_sum = 0.0
            odetime_sum = 0.0
            sampling_time_sum = 0.0
            num_iter = 50
            logging.info('Starting to sample with ODE framework for average NFE calculation...')
            for i in range(num_iter):
                # Note that this is just a quick hack... This sampling will be done 50 times on
                _, nfe, odetime, sampling_time = generate_samples_vada(dae, diffusion_cont, vae, args.batch_size,
                                                                       enable_autocast=args.autocast_eval, ode_eps=eval_args.ode_eps,
                                                                       ode_solver_tol=eval_args.ode_solver_tol, ode_sample=True,
                                                                       prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0)

                nfe_sum = nfe_sum + nfe
                odetime_sum = odetime_sum + odetime
                sampling_time_sum = sampling_time_sum + sampling_time
            logging.info('Sampling NFE (average over {} batches with batchsize {}): {}'
                         .format(num_iter * args.num_process_per_node * args.num_proc_node, args.batch_size, nfe_sum / num_iter))
            logging.info('Sampling Time (average over {} batches with batchsize {}): {} seconds'
                         .format(num_iter * args.num_process_per_node * args.num_proc_node, args.batch_size, sampling_time_sum / num_iter))
            logging.info('ODE Solve Time only (average over {} batches with batchsize {}): {} seconds'
                         .format(num_iter * args.num_process_per_node * args.num_proc_node, args.batch_size, odetime_sum / num_iter))
            writer.add_scalar('ode_sampling_nfe/final_average', nfe_sum / num_iter, 1)

        writer.close()

    elif eval_args.eval_mode == 'sample':
        num_total_samples = eval_args.num_fid_samples
        num_gpus = args.num_process_per_node * args.num_proc_node
        num_sample_per_gpu = int(np.ceil(num_total_samples / num_gpus))
        num_samples = 100
        n = 10
        m = 10
        # n = int(np.floor(np.sqrt(num_samples)))
        num_iter = int(np.ceil(num_sample_per_gpu / num_samples))
        all_nfe = []
        for i in range(num_iter):
            if i == 1:
                start = time()
            if eval_args.ode_sampling:
                logging.info('Starting to sample with ODE framework...')
                logging.info('ODE params: ODE eps %f, ODE tol %f', eval_args.ode_eps, eval_args.ode_solver_tol)
                samples, nfe, odetime, sampling_time = generate_samples_vada(
                    dae, diffusion_cont, vae, num_samples, enable_autocast=args.autocast_eval,
                    ode_eps=eval_args.ode_eps, ode_solver_tol=eval_args.ode_solver_tol, ode_sample=True,
                    prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0, temp=eval_args.temp,
                    vae_temp=eval_args.vae_temp)
            else:
                logging.info('Starting to sample with naive discretization...')
                samples, nfe, odetime, sampling_time = generate_samples_vada(
                    dae, diffusion_disc, vae, num_samples, enable_autocast=args.autocast_eval,
                    prior_var=args.sigma2_max if args.sde_type == 'vesde' else 1.0, temp=eval_args.temp,
                    vae_temp=eval_args.vae_temp)

            logging.info('Sampled new images ({} func. evals, {} seconds for ODE solve, {} seconds for full sampling) of latent space diffusion model.'.format(nfe, odetime, sampling_time))
            all_nfe.append(nfe.cpu().numpy())

            visualize = False
            output_tiled = utils.tile_image(samples, n, m)
            if visualize:
                plt.figure(figsize=(12, 12))
                img = output_tiled.permute(1, 2, 0).float().cpu().numpy()
                if img.shape[2] == 1:
                    img = np.squeeze(img)
                    plt.imshow(img, cmap=plt.get_cmap('gray'))
                else:
                    plt.imshow(img)
                plt.show()
            else:
                # save tiled image
                # save_image(output_tiled, eval_args.save + '/samples.png')
                # logging.info('Saved at: {}'.format(eval_args.save + '/samples.png'))
                file_path = os.path.join(eval_args.save, 'gpu_%d_samples_%d.npz' % (eval_args.global_rank, i))
                np.savez_compressed(file_path, samples=samples.cpu().numpy())
                logging.info('Saved at: {}'.format(file_path))

        if num_iter > 1:
            end = time() - start
            logging.info('timing %f, avg NFE %f' % (end / (num_iter - 1), np.mean(all_nfe)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    # directories for experiment results and checkpoint
    parser.add_argument('--checkpoint', type=str, default='/path/to/checkpoint.pt',
                        help='location of the checkpoint')
    parser.add_argument('--root', type=str, default='/tmp/nvae-diff/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='debug_ode',
                        help='id used for storing intermediate results')
    parser.add_argument('--eval_mode', type=str, default='evaluate', choices=['sample', 'evaluate'],
                        help='evaluation mode. you can choose between sample or evaluate.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--fid_dir', type=str, default='/tmp/nvae-diff/fid-stats',
                        help='path to directory where fid related files are stored')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='The temperature used for sampling.')
    parser.add_argument('--vae_temp', type=float, default=1.0,
                        help='The temperature used for sampling in vae.')
    parser.add_argument('--vae_train_mode', action='store_true', default=False,
                        help='evaluate vae in train mode, suitable for BN experiments.')
    parser.add_argument('--num_iw_samples', type=int, default=1,
                        help='The number of samples from latent space used in IW evaluation.')
    parser.add_argument('--num_iw_inner_samples', type=int, default=1,
                        help='How often we solve the ODE and average when calculating prior probability.')
    parser.add_argument('--num_fid_samples', type=int, default=50000,
                        help='The number of samples used for FID computation.')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
    parser.add_argument('--elbo_eval', action='store_true', default=False,
                        help='if True, we perform discrete ELBO evaluation.')
    parser.add_argument('--fid_disc_eval', action='store_true', default=False,
                        help='if True, we perform FID evaluation.')
    parser.add_argument('--fid_ode_eval', action='store_true', default=False,
                        help='if True, we perform FID evaluation using ODE-based model samples.')
    parser.add_argument('--nll_ode_eval', action='store_true', default=False,
                        help='if True, we perform ODE-based NLL evaluation.')
    parser.add_argument('--nfe_eval', action='store_true', default=False,
                        help='if True, we sample 50 batches of images and average NFEs.')
    parser.add_argument('--ode_sampling', action='store_true', default=False,
                        help='if True, do ODE-based sampling, otherwise regular sampling. Only relevant when sampling.')
    parser.add_argument('--ode_eps', type=float, default=0.00001,
                        help='ODE can only be integrated up to some epsilon > 0.')
    parser.add_argument('--ode_solver_tol', type=float, default=1e-5,
                        help='ODE solver error tolerance.')
    parser.add_argument('--diffusion_steps', type=int, default=0,
                        help='number of diffusion steps')
    # DDP.
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
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