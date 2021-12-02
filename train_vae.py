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

from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from nvae import NVAE
from thirdparty.adamax import Adamax
from util import utils, datasets
from util.sr_utils import SpectralNormCalculator


def main(args):
    # common initialization
    logging, writer = utils.common_init(args.global_rank, args.seed, args.save)

    # Get data loaders.
    train_queue, valid_queue, _ = datasets.get_loaders(args)

    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)
    logging.info('args = %s', args)
    vae = NVAE(args, arch_instance_nvae)
    vae = vae.cuda()

    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))
    # sync all parameters between all gpus by sending param from rank 0 to all gpus.
    utils.broadcast_params(vae.parameters(), args.distributed)

    vae_optimizer = Adamax(vae.parameters(), args.learning_rate_vae,
                           weight_decay=args.weight_decay, eps=1e-3)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min_vae)

    # create SN calculator
    sn_calculator = SpectralNormCalculator()
    sn_calculator.add_conv_layers(vae)
    sn_calculator.add_bn_layers(vae)

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
        vae.load_state_dict(checkpoint['vae_state_dict'])
        vae = vae.cuda()
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
        vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
        if 'sn_calculator' in checkpoint:   # for backward compatibility
            sn_calculator.load_state_dict(checkpoint['sn_calculator'], torch.device("cuda"))
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        global_step = checkpoint['global_step']
        best_score = checkpoint['best_score']
        logging.info('loaded the model at epoch %d.', init_epoch)
    else:
        global_step, epoch, init_epoch, best_score = 0, 0, 0, 1e10

    # The milestones at which we save additional checkpoints from intermediate epochs.
    milestones = [0.5, 0.7]
    milestones_epochs = set([int(args.epochs * m) for m in milestones])

    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            vae_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)
        train_obj, global_step = train_vae(args, train_queue, vae, vae_optimizer, grad_scalar, global_step, warmup_iters,
                                           writer, logging, sn_calculator)

        logging.info('train_loss %f', train_obj)
        writer.add_scalar('train/loss_epoch', train_obj, global_step)

        # TODO: define a save model frequency different than evaluation frequency.
        # generate samples less frequently
        num_evaluations = 40
        eval_freq = max(args.epochs // num_evaluations, 1)
        if ((epoch + 1) % eval_freq == 0 or epoch == (args.epochs - 1)):
            vae.eval()
            n = int(np.floor(np.sqrt(min(16, args.batch_size))))
            num_samples = n ** 2
            for t in [0.7, 1.0]:
                output_img = vae.sample(num_samples, t, enable_autocast=args.autocast_eval)
                output_tiled = utils.tile_image(output_img, n)
                writer.add_image('generated_%0.1f' % t, output_tiled, global_step)

            valid_neg_log_p, valid_nelbo = test_vae(valid_queue, vae, num_samples=10, args=args, logging=logging)
            current_score = valid_neg_log_p
            logging.info('valid bpd nelbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)
            writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch)
            writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch)
            writer.add_scalar('val/nat_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nat_elbo', valid_nelbo, epoch)

            if args.global_rank == 0 and current_score < best_score:
                best_score = current_score
                logging.info('saving the model.')
                content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                           'grad_scalar': grad_scalar.state_dict(), 'best_score': best_score,
                           'vae_state_dict': vae.state_dict(), 'vae_optimizer': vae_optimizer.state_dict(),
                           'vae_scheduler': vae_scheduler.state_dict(), 'sn_calculator': sn_calculator.state_dict()}
                torch.save(content, checkpoint_file)

        # save snapshots of model at certain training milestones.
        if args.global_rank == 0 and epoch in milestones_epochs:
            checkpoint_file_ms = os.path.join(args.save, 'checkpoint_epoch_%d.pt' % epoch)
            logging.info('saving the model at milestone epochs.')
            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                       'grad_scalar': grad_scalar.state_dict(), 'best_score': best_score,
                       'vae_state_dict': vae.state_dict(), 'vae_optimizer': vae_optimizer.state_dict(),
                       'vae_scheduler': vae_scheduler.state_dict(), 'sn_calculator': sn_calculator.state_dict()}
            torch.save(content, checkpoint_file_ms)


    # loading the model at the best score
    # make all nodes wait for rank 0 to finish saving the files
    if args.distributed:
        dist.barrier()

    # Final validation
    valid_neg_log_p, valid_nelbo = test_vae(valid_queue, vae, num_samples=1000, args=args, logging=logging)
    logging.info('final valid bpd nelbo %f', valid_nelbo * bpd_coeff)
    logging.info('final valid bpd neg log p %f', valid_neg_log_p * bpd_coeff)
    writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch + 1)
    writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch + 1)
    writer.add_scalar('val/nat_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nat_elbo', valid_nelbo, epoch + 1)
    writer.close()


def train_vae(args, train_queue, model, optimizer, grad_scalar, global_step, warmup_iters, writer, logging, sn_calculator):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()
    for step, x in enumerate(train_queue):
        x = utils.common_x_operations(x, args.num_x_bits)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate_vae * float(global_step) / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()
        with autocast(enabled=args.autocast_train):
            logits, all_log_q, all_eps = model(x)
            log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
            output = model.decoder_output(logits)
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff,
                                      args.kl_max_coeff)

            recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = sn_calculator.spectral_norm_parallel()
            bn_loss = sn_calculator.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            if (global_step + 1) % 10000 == 0:  # reduced frequency
                n = int(np.floor(np.sqrt(x.size(0))))
                x_img = x[:n*n]
                output_img = output.mean()
                output_img = output_img[:n*n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                in_out_tiled = utils.unsymmetrize_image_data(in_out_tiled)
                writer.add_image('reconstruction', in_out_tiled, global_step)

            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)
            writer.add_scalar('train/recon_iter', torch.mean(
                utils.reconstruction_loss(output, x, crop=model.crop_output)), global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test_vae(valid_queue, model, num_samples, args, logging):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, x in enumerate(valid_queue):
        x = utils.common_x_operations(x, args.num_x_bits)
        with torch.no_grad():
            nelbo, log_iw = [], []
            for k in range(num_samples):
                logits, all_log_q, all_eps = model(x)
                log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
                output = model.decoder_output(logits)
                recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss + balanced_kl
                nelbo.append(nelbo_batch)
                log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg


def create_generator_vae(model, batch_size, num_total_samples, enable_autocast):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            image = model.sample(batch_size, 1.0, None, enable_autocast)
        yield image.float()


def test_vae_fid(model, args, total_fid_samples):
    dims = 2048
    device = 'cuda'
    enable_autocast = args.autocast_eval
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae(model, args.batch_size, num_sample_per_gpu, enable_autocast)
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


def infer_active_variables(train_queue, vae, args, max_iter=None):
    kl_meter = utils.AvgrageMeter()
    vae.eval()
    for step, x in enumerate(train_queue):
        if max_iter is not None and step > max_iter:
            break

        x = utils.common_x_operations(x, args.num_x_bits)
        with autocast(enabled=args.autocast_train):
            # apply vae:
            with torch.set_grad_enabled(False):
                _, all_log_q, all_eps = vae(x)
                all_eps = vae.concat_eps_per_scale(all_eps)
                all_log_q = vae.concat_eps_per_scale(all_log_q)
                log_q, log_p, kl_all, kl_diag = utils.vae_terms(all_log_q, all_eps)
                kl_meter.update(kl_diag[0], 1)  # only the top scale

    utils.average_tensor(kl_meter.avg, args.distributed)
    return kl_meter.avg > 0.1

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
    parser.add_argument('--learning_rate_vae', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min_vae', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    parser.add_argument('--kl_max_coeff', type=float, default=1.,
                        help='The constant value used for max KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--log_sig_q_scale', type=float, default=5.,        # we used to use [-5, 5]
                        help='log sigma q is clamped into [-log_sig_q_scale, log_sig_q_scale].')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--channel_mult', nargs='+', type=int,
                        help='channel multiplier per scale', )
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--decoder_dist', type=str, default='dml', choices=['normal', 'dml', 'dl', 'bin'],
                        help='Distribution used in VAE decoder: Normal, Discretized Mix of Logistic,'
                             'Bernoulli, or discretized logistic.')
    parser.add_argument('--progressive_input_vae', type=str, default='none', choices=['none', 'input_skip'],
                        help='progressive type for input')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
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
