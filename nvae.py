# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from layers.neural_operations import EncCombinerCell, DecCombinerCell, Conv2D, InputSkipCombinerCell, BNSwishConv, UpSample
from layers.cells import Cell, PairedCellAR

from util.utils import get_input_size, groups_per_scale, soft_clamp5, soft_clamp, unsymmetrize_image_data
from util.distributions import Normal, DiscMixLogistic, DiscLogistic, Bernoulli


class NVAE(nn.Module):
    def __init__(self, args, arch_instance):
        super(NVAE, self).__init__()
        self.arch_instance = arch_instance
        self.dataset = args.dataset
        self.crop_output = self.dataset in {'mnist', 'omniglot'}
        self.log_sig_q_scale = torch.Tensor([args.log_sig_q_scale]).cuda()
        self.num_bits = args.num_x_bits
        self.decoder_dist = args.decoder_dist

        self.num_latent_scales = args.num_latent_scales         # number of spatial scales that latent layers will reside
        self.num_groups_per_scale = args.num_groups_per_scale   # number of groups of latent vars. per scale
        self.num_latent_per_group = args.num_latent_per_group   # number of latent vars. per group
        self.groups_per_scale = groups_per_scale(self.num_latent_scales, self.num_groups_per_scale)

        # encoder parameteres
        self.num_channels_enc = args.num_channels_enc
        self.num_preprocess_blocks = args.num_preprocess_blocks  # block is defined as series of Normal followed by Down
        self.num_preprocess_cells = args.num_preprocess_cells    # number of cells per block
        self.num_cell_per_cond_enc = args.num_cell_per_cond_enc  # number of cell for each conditional in encoder

        # decoder parameters
        self.num_channels_dec = args.num_channels_dec
        self.num_postprocess_blocks = args.num_postprocess_blocks
        self.num_postprocess_cells = args.num_postprocess_cells
        self.num_cell_per_cond_dec = args.num_cell_per_cond_dec  # number of cell for each conditional in decoder

        # progressive input
        self.progressive_input = args.progressive_input_vae

        # general cell parameters
        self.input_size = get_input_size(self.dataset)

        # decoder param
        self.num_mix_output = 10

        total_scales = self.num_preprocess_blocks + self.num_latent_scales
        self.channels_mult = args.channel_mult
        assert self.num_preprocess_blocks == self.num_postprocess_blocks
        assert len(self.channels_mult) == total_scales

        # used for generative purpose
        c_scaling = self.channels_mult[-1]
        spatial_scaling = 2 ** (self.num_preprocess_blocks + self.num_latent_scales - 1)
        prior_ftr0_size = (int(c_scaling * self.num_channels_dec), self.input_size // spatial_scaling,
                           self.input_size // spatial_scaling)
        self.prior_ftr0 = nn.Parameter(torch.rand(size=prior_ftr0_size), requires_grad=True)

        self.stem = self.init_stem()
        self.pre_process, scale_ind = self.init_pre_process(scale_ind=0)

        self.enc_tower, self.progressive_input_cells, scale_ind = self.init_encoder_tower(scale_ind)

        self.with_nf = args.num_nf > 0
        self.num_flows = args.num_nf

        self.enc_sampler, self.dec_sampler, self.nf_cells, self.eps_conv = self.init_normal_sampler(scale_ind)

        self.dec_tower, scale_ind = self.init_decoder_tower(scale_ind)

        self.post_process, scale_ind = self.init_post_process(scale_ind)

        self.image_conditional = self.init_image_conditional(scale_ind)

    def init_stem(self):
        Cout = self.num_channels_enc
        Cin = 1 if self.dataset in {'mnist', 'omniglot'} else 3
        stem = Conv2D(Cin, Cout, 3, padding=1, bias=True)
        return stem

    def init_pre_process(self, scale_ind):
        pre_process = nn.ModuleList()
        for b in range(self.num_preprocess_blocks):
            for c in range(self.num_preprocess_cells):
                if c == self.num_preprocess_cells - 1:  # and b < self.num_preprocess_blocks - 1:
                    arch = self.arch_instance['down_pre']
                    num_ci = int(self.num_channels_enc * self.channels_mult[scale_ind])
                    num_co = int(self.num_channels_enc * self.channels_mult[scale_ind + 1])
                    cell = Cell(num_ci, num_co, cell_type='down_pre', arch=arch)
                    scale_ind += 1
                else:
                    arch = self.arch_instance['normal_pre']
                    num_c = int(self.num_channels_enc * self.channels_mult[scale_ind])
                    cell = Cell(num_c, num_c, cell_type='normal_pre', arch=arch)

                pre_process.append(cell)

        return pre_process, scale_ind

    def init_encoder_tower(self, scale_ind):
        enc_tower, progressive_input_cells = nn.ModuleList(), nn.ModuleList()
        for s in range(self.num_latent_scales):
            # number of channels at the input of scale
            num_ci_scale = int(self.num_channels_enc * self.channels_mult[scale_ind])
            for g in range(self.groups_per_scale[s]):
                for c in range(self.num_cell_per_cond_enc):
                    arch = self.arch_instance['normal_enc']
                    num_c = int(self.num_channels_enc * self.channels_mult[scale_ind])
                    cell = Cell(num_c, num_c, cell_type='normal_enc', arch=arch)
                    enc_tower.append(cell)

                # add encoder combiner
                if not (s == self.num_latent_scales - 1 and g == self.groups_per_scale[s] - 1):
                    num_ce = int(self.num_channels_enc * self.channels_mult[scale_ind])
                    num_cd = int(self.num_channels_dec * self.channels_mult[scale_ind])
                    cell = EncCombinerCell(num_ce, num_cd, num_ce, cell_type='combiner_enc')
                    enc_tower.append(cell)

            # down cells after finishing a scale
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['down_enc']
                num_ci = int(self.num_channels_enc * self.channels_mult[scale_ind])
                num_co = int(self.num_channels_enc * self.channels_mult[scale_ind + 1])
                cell = Cell(num_ci, num_co, cell_type='down_enc', arch=arch)
                enc_tower.append(cell)
                scale_ind += 1

                # apply progressive input after downsampling
                if self.progressive_input == 'input_skip':
                    progressive_input_cells.append(InputSkipCombinerCell(num_co, num_ci_scale, num_co, cell_type='combiner_enc'))

        return enc_tower, progressive_input_cells, scale_ind

    def init_normal_sampler(self, scale_ind):
        enc_sampler, dec_sampler, nf_cells, eps_conv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                # build mu, sigma generator for encoder
                num_c = int(self.num_channels_enc * self.channels_mult[scale_ind])
                cell = nn.Sequential(
                    nn.ELU(),
                    Conv2D(num_c, 2 * self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
                )
                enc_sampler.append(cell)
                # build NF
                for n in range(self.num_flows):
                    arch = self.arch_instance['ar_nn']
                    num_c1 = int(self.num_channels_enc * self.channels_mult[scale_ind])
                    num_c2 = 8 * self.num_latent_per_group  # use 8x features
                    nf_cells.append(PairedCellAR(self.num_latent_per_group, num_c1, num_c2, arch))

                cell = Conv2D(self.num_latent_per_group, self.num_latent_per_group, kernel_size=3, padding=1, bias=True)
                eps_conv.append(cell)

            scale_ind -= 1

        return enc_sampler, dec_sampler, nf_cells, eps_conv

    def init_decoder_tower(self, scale_ind):
        # create decoder tower
        dec_tower = nn.ModuleList()
        for s in range(self.num_latent_scales):
            for g in range(self.groups_per_scale[self.num_latent_scales - s - 1]):
                # add decoder combiner (+)
                num_c = int(self.num_channels_dec * self.channels_mult[scale_ind])
                cell = DecCombinerCell(num_c, self.num_latent_per_group, num_c, cell_type='combiner_dec')
                dec_tower.append(cell)

                # add residual cells per conditional <r>
                for c in range(self.num_cell_per_cond_dec):
                    arch = self.arch_instance['normal_dec']
                    cell = Cell(num_c, num_c, cell_type='normal_dec', arch=arch)
                    dec_tower.append(cell)

            # up cells after finishing a scale
            if s < self.num_latent_scales - 1:
                arch = self.arch_instance['up_dec']
                num_ci = int(self.num_channels_dec * self.channels_mult[scale_ind])
                num_co = int(self.num_channels_dec * self.channels_mult[scale_ind-1])
                cell = Cell(num_ci, num_co, cell_type='up_dec', arch=arch)
                dec_tower.append(cell)
                scale_ind -= 1

        return dec_tower, scale_ind

    def init_post_process(self, scale_ind):
        post_process = nn.ModuleList()
        for b in range(self.num_postprocess_blocks):
            for c in range(self.num_postprocess_cells):
                # Upsampling cell at the beginning of the block.
                # No Upsampling in the first block
                if c == 0:  # and b > 0:
                    arch = self.arch_instance['up_post']
                    num_ci = int(self.num_channels_dec * self.channels_mult[scale_ind])
                    num_co = int(self.num_channels_dec * self.channels_mult[scale_ind-1])
                    cell = Cell(num_ci, num_co, cell_type='up_post', arch=arch)
                    scale_ind -= 1
                else:
                    arch = self.arch_instance['normal_post']
                    num_c = int(self.num_channels_dec * self.channels_mult[scale_ind])
                    cell = Cell(num_c, num_c, cell_type='normal_post', arch=arch)

                post_process.append(cell)

        return post_process, scale_ind

    def init_image_conditional(self, scale_ind):
        C_in = int(self.num_channels_dec * self.channels_mult[scale_ind])
        if self.decoder_dist in {'normal', 'dl'}:
            C_out = 6
        elif self.decoder_dist in {'bin'}:
            C_out = 1
        elif self.decoder_dist in {'dml'}:
            C_out = 10 * self.num_mix_output
        else:
            raise NotImplementedError

        return nn.Sequential(nn.ELU(),
                             Conv2D(C_in, C_out, 3, padding=1, bias=True))

    def forward(self, x):
        s = self.stem(x)

        # perform pre-processing
        for cell in self.pre_process:
            s = cell(s)

        if self.progressive_input == 'input_skip':
            input_pyramid = s
            progressive_input_index = 0

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)
                # apply progressive input after downsampling
                if cell.cell_type == 'down_enc' and self.progressive_input == 'input_skip':
                    # update input_pyramid and mix it with s
                    s, input_pyramid = self.progressive_input_cells[progressive_input_index](s, input_pyramid)
                    progressive_input_index += 1

        # reverse combiner cells and their input for decoder
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec, nf_offset = 0, 0
        all_q, all_log_q, all_eps = [], [], []
        ftr_enc = s
        batch_size = s.shape[0]
        s = self.prior_ftr0.unsqueeze(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.dec_tower:
            if cell.cell_type == 'combiner_dec':
                # form the encoder
                if idx_dec > 0:
                    ftr_enc = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)

                param = self.enc_sampler[idx_dec](ftr_enc)
                mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                mu_q, log_sig_q = soft_clamp5(mu_q), soft_clamp(log_sig_q, self.log_sig_q_scale)
                dist = Normal(mu_q, log_sig_q)
                eps, _ = dist.sample()
                log_q_conv = dist.log_p(eps)
                # apply NF
                for n in range(self.num_flows):
                    eps, log_det = self.nf_cells[nf_offset + n](eps, ftr_enc)
                    log_q_conv -= log_det

                nf_offset += self.num_flows
                all_log_q.append(log_q_conv)
                all_q.append(dist)
                all_eps.append(eps)

                z = self.eps_conv[idx_dec](eps)
                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                # main decoder tower
                s = cell(s)

        for cell in self.post_process:
            s = cell(s)

        logits = self.image_conditional(s)

        return logits, all_log_q, all_eps

    def sample(self, num_samples, t, eps_z=None, enable_autocast=False):
        with torch.no_grad():
            with autocast(enable_autocast):
                num_eps_z_given = len(eps_z) if eps_z is not None else 0

                # z = self.eps_conv[0](eps)
                s = self.prior_ftr0.unsqueeze(0)
                s = s.expand(num_samples, -1, -1, -1)
                idx_dec = 0
                for cell in self.dec_tower:
                    if cell.cell_type == 'combiner_dec':
                        if idx_dec < num_eps_z_given:
                            eps = eps_z[idx_dec]
                        else:
                            b, _, h, w = s.shape
                            size = [b, self.num_latent_per_group, h, w]
                            dist = Normal(mu=torch.zeros(size=size, device='cuda'),
                                          log_sigma=torch.zeros(size=size, device='cuda'))
                            eps, _ = dist.sample(t=t)

                        z = self.eps_conv[idx_dec](eps)
                        s = cell(s, z)
                        idx_dec += 1
                    else:
                        # main decoder tower
                        s = cell(s)

                for cell in self.post_process:
                    s = cell(s)

                logits = self.image_conditional(s)

                output = self.decoder_output(logits)
                output_img = output.mean()
                output_img = output_img.clamp(min=-1., max=1.)
                output_img = unsymmetrize_image_data(output_img)
        return output_img

    def decoder_output(self, logits):
        if self.decoder_dist in {'normal'}:
            logits = soft_clamp5(logits)
            mu, log_sigma = torch.chunk(logits, 2, dim=1)
            return Normal(mu, log_sigma)
        elif self.decoder_dist == 'dml':
            return DiscMixLogistic(logits, self.num_mix_output, num_bits=self.num_bits)
        elif self.decoder_dist == 'dl':
            assert self.num_bits == 8, 'we have not implemented low precision'
            return DiscLogistic(logits)
        elif self.decoder_dist == 'bin':
            assert self.num_bits == 8, 'changing the number of bits have no effect on binary datasets.'
            return Bernoulli(logits=logits)
        else:
            raise NotImplementedError

    def latent_structure(self):
        structure = [self.num_groups_per_scale * self.num_latent_per_group] * self.num_latent_scales
        return structure

    def decompose_eps(self, eps):
        # assume eps is from the top scale
        eps_z = torch.chunk(eps, self.num_groups_per_scale, dim=1)
        return list(eps_z)

    def concat_eps_per_scale(self, all_eps):
        concat_eps = []
        offset = 0
        for s in range(self.num_latent_scales):
            num_group = self.groups_per_scale[self.num_latent_scales - s - 1]
            concat_eps.append(torch.cat(all_eps[offset:offset + num_group], dim=1))
            offset += num_group

        return concat_eps
