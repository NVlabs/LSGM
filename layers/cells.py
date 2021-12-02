# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from util.utils import get_stride_for_cell_type
from layers.neural_operations import get_skip_connection, SE, OPS
from layers.neural_ar_operations import ELUConv as ARELUConv
from layers.neural_ar_operations import ARInvertedResidual, MixLogCDFParam, mix_log_cdf_flow
from torch.cuda.amp import autocast


class Cell(nn.Module):
    def __init__(self, Cin, Cout, cell_type, arch, temb_in=0, dropout_p=0., apply_attn=False):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        stride = get_stride_for_cell_type(self.cell_type)
        # skip connection
        self.skip = get_skip_connection(Cin, Cout, stride)

        # conv branch
        conv_branch = arch['conv_branch']
        self._num_nodes = len(conv_branch)
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            C = Cin if i == 0 else Cout
            primitive = conv_branch[i]
            op = OPS[primitive](C, Cout, stride, dropout_p)
            self._ops.append(op)

        # SE
        self.use_se = arch['se']
        if self.use_se:
            self.se = SE(Cout, Cout)

        # Attention
        # self.use_attention = arch.get('attn_type', None) is not None
        self.use_attention = apply_attn
        if self.use_attention:
            self.attn = OPS['attn'](Cout, Cout, stride, dropout_p)

        # Time embedding
        self.use_temb = temb_in > 0
        if self.use_temb:
            hidden_dim = self._ops[0].hidden_dim
            self.temb_proj = nn.Linear(temb_in, hidden_dim)

        self.apply_sqrt2 = arch.get('apply_sqrt2', False)

    def forward(self, s, temb=None):
        # skip branch
        skip = self.skip(s)

        # conv branch
        for i in range(self._num_nodes):
            # apply temb after the first block
            if i == 0 and self.use_temb:
                # view in Linear has a bug that causes OOM:
                # https://github.com/pytorch/pytorch/pull/48696
                with autocast(False):
                    temb = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1)
            else:
                temb = None
            s = self._ops[i](s, temb)

        # SE
        s = self.se(s) if self.use_se else s

        # residual cell
        if self.apply_sqrt2:
            out = (skip + s) / 1.414213  # np.sqrt(2)
        else:
            out = skip + 0.1 * s         # vae

        # apply attention after residual cell
        if self.use_attention:
            out = out + 0.1 * self.attn(out)

        return out


class CellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch, mirror):
        super(CellAR, self).__init__()
        assert num_c % num_z == 0

        self.cell_type = 'ar_nn'

        # s0 will the random samples
        ex = 6
        self.conv = ARInvertedResidual(num_z, num_ftr, ex=ex, mirror=mirror)

        self.use_mix_log_cdf = False
        if self.use_mix_log_cdf:
            self.param = MixLogCDFParam(num_z, num_mix=3, num_ftr=self.conv.hidden_dim, mirror=mirror)
        else:
            # 0.1 helps bring mu closer to 0 initially
            self.mu = ARELUConv(self.conv.hidden_dim, num_z, kernel_size=1, padding=0, masked=True, zero_diag=False,
                                weight_init_coeff=0.1, mirror=mirror)

    def forward(self, z, ftr):
        s = self.conv(z, ftr)

        if self.use_mix_log_cdf:
            logit_pi, mu, log_s, log_a, b = self.param(s)
            new_z, log_det = mix_log_cdf_flow(z, logit_pi, mu, log_s, log_a, b)
        else:
            mu = self.mu(s)
            new_z = (z - mu)
            log_det = torch.zeros_like(new_z)

        return new_z, log_det


class PairedCellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch=None):
        super(PairedCellAR, self).__init__()
        self.cell1 = CellAR(num_z, num_ftr, num_c, arch, mirror=False)
        self.cell2 = CellAR(num_z, num_ftr, num_c, arch, mirror=True)

    def forward(self, z, ftr):
        new_z, log_det1 = self.cell1(z, ftr)
        new_z, log_det2 = self.cell2(new_z, ftr)

        log_det1 += log_det2
        return new_z, log_det1