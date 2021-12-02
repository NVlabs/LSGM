# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.multiprocessing import Process

from nvae import NVAE
from train_vae import test_vae, test_vae_fid
from util import utils, datasets


def main(eval_args):
    # common initialization
    logging, writer = utils.common_init(eval_args.global_rank, eval_args.seed, eval_args.save)

    # load a checkpoint
    logging.info('#' * 80)
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    if not hasattr(args, 'num_x_bits'):
        logging.info('*** Setting %s manually ****', 'num_x_bits')
        setattr(args, 'num_x_bits', 8)

    if not hasattr(args, 'channel_mult'):
        logging.info('*** Setting %s manually ****', 'channel_mult')
        setattr(args, 'channel_mult', [1, 2])

    epoch = checkpoint['epoch']
    logging.info('loaded the model at epoch %d', checkpoint['epoch'])

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)
    logging.info('args = %s', args)

    # load VAE
    vae = NVAE(args, arch_instance_nvae)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae = vae.cuda()
    logging.info('VAE: param size = %fM ', utils.count_parameters_in_M(vae))

    # replace a few fields in args based on eval_args
    # this will allow train/evaluate on different systems
    args.num_proc_node = eval_args.num_proc_node
    args.num_process_per_node = eval_args.num_process_per_node
    args.data = eval_args.data
    if eval_args.batch_size > 0:
        args.batch_size = eval_args.batch_size

    if eval_args.eval_mode == 'evaluate':
        # replace a few fields in args based on eval_args
        # this will allow train/evaluate on different systems
        args.num_proc_node = eval_args.num_proc_node
        args.num_process_per_node = eval_args.num_process_per_node
        args.data = eval_args.data
        if eval_args.batch_size > 0:
            args.batch_size = eval_args.batch_size

        if eval_args.nll_eval:
            # load train valid queue
            bpd_coeff = utils.get_bpd_coeff(args.dataset)
            train_queue, valid_queue, num_classes = datasets.get_loaders(args)

            if eval_args.eval_on_train:
                logging.info('Using the training data for eval.')
                valid_queue = train_queue

            neg_log_p, nelbo = test_vae(valid_queue, vae, eval_args.num_iw_samples, args, logging)
            logging.info('valid bpd nelbo %f', nelbo * bpd_coeff)
            logging.info('valid bpd neg log p %f', neg_log_p * bpd_coeff)
            logging.info('valid nat nelbo %f', nelbo)
            logging.info('valid nat neg log p %f', neg_log_p)

        if eval_args.fid_eval:
            args.fid_dir = eval_args.fid_dir
            num_fid_samples = 50000
            logging.info('Running FID evaluation...')
            fid = test_vae_fid(vae, args, num_fid_samples)
            logging.info('valid FID: {}'.format(fid))

    elif eval_args.eval_mode == 'sample':
        vae.eval()
        n = 3
        m = 5
        num_samples = n * m

        output_img = vae.sample(num_samples=num_samples, t=1.)
        output_tiled = utils.tile_image(output_img, n, m)

        plt.rcParams['figure.figsize'] = (12, 12)
        plt.imshow(output_tiled.cpu().permute(1, 2, 0).numpy())
        save_image(output_tiled, eval_args.save + '/vae_samples.png')
        logging.info('Saved at: {}'.format(eval_args.save + '/vae_samples.png'))



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
    parser.add_argument('--nll_eval', action='store_true', default=False,
                        help='if True, we perform NLL evaluation.')
    parser.add_argument('--fid_eval', action='store_true', default=False,
                        help='if True, we perform FID evaluation.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--fid_dir', type=str, default='/tmp/nvae-diff/fid-stats',
                        help='A dir to store fid related files')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='The temperature used for sampling.')
    parser.add_argument('--num_iw_samples', type=int, default=1,
                        help='The number of samples from latent space used in IW evaluation.')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
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