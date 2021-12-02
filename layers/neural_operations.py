# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from thirdparty.swish import Swish as SwishFN
from thirdparty.inplaced_sync_batchnorm import SyncBatchNormSwish
from thirdparty.checkpoint import checkpoint

from util.utils import average_tensor
from collections import OrderedDict

BN_EPS = 1e-5
SYNC_BN = True

OPS = OrderedDict([
    ('res_bnswish', lambda Cin, Cout, stride, dropout: BNSwishConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish_x2', lambda Cin, Cout, stride, dropout: BNSwishConvX2(Cin, Cout, 3, stride, 1)),
    ('res_gnswish_x2', lambda Cin, Cout, stride, dropout: GNSwishConv(Cin, Cout, 3, stride, 1, 1, dropout)),
    ('mconv_e6k5g0', lambda Cin, Cout, stride, dropout: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=5, g=0)),
    ('mconv_e3k5g0', lambda Cin, Cout, stride, dropout: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=0)),
    ('mconv_e6k5g0_gn', lambda Cin, Cout, stride, dropout: InvertedResidualGN(Cin, Cout, stride, ex=6, dil=1, k=5, g=0)),
    ('attn', lambda Cin, Cout, stride, dropout: Attention(Cin))
])


def get_skip_connection(Cin, Cout, stride):
    if stride == 1:
        return Identity()
    elif stride == 2:
        return FactorizedReduce(Cin, Cout)
    elif stride == -1:
        return nn.Sequential(UpSample(), Conv2D(Cin, Cout, kernel_size=1))


def norm(t, dim):
    return torch.sqrt(torch.sum(t * t, dim))


def logit(t):
    return torch.log(t) - torch.log(1 - t)


def act(t):
    # The following implementation has lower memory.
    return SwishFN.apply(t)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return act(x)

@torch.jit.script
def normalize_weight_jit(log_weight_norm, weight):
    n = torch.exp(log_weight_norm)
    wn = torch.sqrt(torch.sum(weight * weight, dim=[1, 2, 3]))   # norm(w)
    weight = n * weight / (wn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-5)
    return weight


class Conv2D(nn.Conv2d):
    """Allows for weights as input."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, data_init=False,
                 weight_norm=True):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(Conv2D, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)

        self.log_weight_norm = None
        if weight_norm:
            init = norm(self.weight, dim=[1, 2, 3]).view(-1, 1, 1, 1)
            self.log_weight_norm = nn.Parameter(torch.log(init + 1e-2), requires_grad=True)

        self.data_init = data_init
        self.init_done = False
        self.weight_normalized = self.normalize_weight()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W).
            params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
        """
        # do data based initialization
        if self.data_init and not self.init_done:
            with torch.no_grad():
                weight = self.weight / (norm(self.weight, dim=[1, 2, 3]).view(-1, 1, 1, 1) + 1e-5)
                bias = None
                out = F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
                mn = torch.mean(out, dim=[0, 2, 3])
                st = 5 * torch.std(out, dim=[0, 2, 3])

                # get mn and st from other GPUs
                average_tensor(mn, is_distributed=True)
                average_tensor(st, is_distributed=True)

                if self.bias is not None:
                    self.bias.data = - mn / (st + 1e-5)
                self.log_weight_norm.data = -torch.log((st.view(-1, 1, 1, 1) + 1e-5))
                self.init_done = True

        self.weight_normalized = self.normalize_weight()

        bias = self.bias
        return F.conv2d(x, self.weight_normalized, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def normalize_weight(self):
        """ applies weight normalization """
        if self.log_weight_norm is not None:
            weight = normalize_weight_jit(self.log_weight_norm, self.weight)
        else:
            weight = self.weight

        return weight


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SyncBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SyncBatchNorm, self).__init__()
        self.bn = nn.SyncBatchNorm(*args, **kwargs)

    def forward(self, x):
        # Sync BN only works with distributed data parallel with 1 GPU per process. I don't use DDP, so I need to let
        # Sync BN to know that I have 1 gpu per process.
        self.bn.ddp_gpu_size = 1
        return self.bn(x)


class GroupNormSwish(nn.Module):
    def __init__(self, C_in, eps=BN_EPS, checkpointing=False):  #  checkpointing=True
        super(GroupNormSwish, self).__init__()
        self.bn_swish = nn.Sequential(get_groupnorm(C_in, eps),
                                      Swish())
        self.checkpointing = checkpointing

    def forward(self, x):
        if self.checkpointing:
            return checkpoint(self.bn_swish, x, preserve_rng_state=False)
        else:
            return self.bn_swish(x)


# quick switch between multi-gpu, single-gpu batch norm
def get_batchnorm(C_in, eps=BN_EPS, momentum=0.05):
    if SYNC_BN:
        return SyncBatchNorm(C_in, eps, momentum)
    else:
        return nn.BatchNorm2d(C_in, eps, momentum)


def get_groupnorm(C_in, eps=BN_EPS):
    num_c_per_group = 16
    assert C_in % num_c_per_group == 0
    return nn.GroupNorm(num_groups=C_in // num_c_per_group, num_channels=C_in, eps=eps)


class BNSwishConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNSwishConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn_act = SyncBatchNormSwish(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x, temb=None):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = self.bn_act(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNSwishConvX2(nn.Module):
    """ Same as BNSwishConv but it applies two convs in a row. """
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNSwishConvX2, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn_act_0 = SyncBatchNormSwish(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation,
                             weight_norm=False)
        self.bn_act_1 = SyncBatchNormSwish(C_out, eps=BN_EPS, momentum=0.05)
        self.conv_1 = Conv2D(C_out, C_out, kernel_size, stride=1, padding=padding, bias=True, dilation=dilation,
                             weight_norm=True)

    def forward(self, x, temb=None):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = self.bn_act_0(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        out = self.bn_act_1(out)
        out = self.conv_1(out)
        return out


class GNSwishConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, dropout_prop=0.):
        super(GNSwishConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.dropout = nn.Dropout(p=dropout_prop) if dropout_prop > 0. else None
        self.hidden_dim = C_out
        self.gn_act0 = GroupNormSwish(C_in, eps=BN_EPS)
        self.conv0 = Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation, weight_norm=False)
        self.gn_act1 = GroupNormSwish(C_out, eps=BN_EPS)
        self.conv1 = Conv2D(C_out, C_out, kernel_size, stride=1, padding=padding, bias=True, dilation=dilation, weight_norm=False)

    def forward(self, x, temb=None):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        # apply first gn + act + conv
        out = self.gn_act0(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv0(out)

        if temb is not None:
            out += temb

        # apply second gn + act + conv
        out = self.gn_act1(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv1(out)

        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.conv_1 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_2 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_3 = Conv2D(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_4 = Conv2D(C_in, C_out - 3 * (C_out // 4), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = act(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class EncCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(EncCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2D(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


class InputSkipCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(InputSkipCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = BNSwishConv(Cin2, Cout, kernel_size=3, stride=2, padding=1)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out, x2


# original combiner
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(DecCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2D(Cin1 + Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


class DenoisingDecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(DenoisingDecCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2D(Cin1 + Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


class ConvBNSwish(nn.Module):
    def __init__(self, Cin, Cout, k=3, stride=1, groups=1, dilation=1):
        padding = dilation * (k - 1) // 2
        super(ConvBNSwish, self).__init__()

        self.conv = nn.Sequential(
            Conv2D(Cin, Cout, k, stride, padding, groups=groups, bias=False, dilation=dilation, weight_norm=False),
            SyncBatchNormSwish(Cout, eps=BN_EPS, momentum=0.05)  # drop in replacement for BN + Swish
        )

    def forward(self, x):
        return self.conv(x)


class ConvGNSwish(nn.Module):
    def __init__(self, Cin, Cout, k=3, stride=1, groups=1, dilation=1):
        padding = dilation * (k - 1) // 2
        super(ConvGNSwish, self).__init__()

        self.conv = nn.Sequential(
            Conv2D(Cin, Cout, k, stride, padding, groups=groups, bias=False, dilation=dilation, weight_norm=False),
            GroupNormSwish(Cout, eps=BN_EPS)
        )

    def forward(self, x):
        return self.conv(x)


class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        # view in Linear has a bug that causes OOM:
        # https://github.com/pytorch/pytorch/pull/48696
        with autocast(False):
            se = se.float()
            se = se.view(se.size(0), -1)
            se = self.se(se)
            se = se.view(se.size(0), -1, 1, 1)
        return x * se


class InvertedResidual(nn.Module):
    def __init__(self, Cin, Cout, stride, ex, dil, k, g):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2, -1]

        hidden_dim = int(round(Cin * ex))
        self.use_res_connect = self.stride == 1 and Cin == Cout
        self.upsample = self.stride == -1
        self.stride = abs(self.stride)
        groups = hidden_dim if g == 0 else g

        layers0 = [nn.UpsamplingNearest2d(scale_factor=2)] if self.upsample else []
        layers = [get_batchnorm(Cin, eps=BN_EPS, momentum=0.05),
                  ConvBNSwish(Cin, hidden_dim, k=1),
                  ConvBNSwish(hidden_dim, hidden_dim, stride=self.stride, groups=groups, k=k, dilation=dil),
                  Conv2D(hidden_dim, Cout, 1, 1, 0, bias=False, weight_norm=False),
                  get_batchnorm(Cout, momentum=0.05)]

        layers0.extend(layers)
        self.conv = nn.Sequential(*layers0)

    def forward(self, x, temb=None):
        return self.conv(x)


class InvertedResidualGN(nn.Module):
    def __init__(self, Cin, Cout, stride, ex, dil, k, g):
        super(InvertedResidualGN, self).__init__()
        self.stride = stride
        assert stride in [1, 2, -1]

        self.hidden_dim = int(round(Cin * ex))
        self.use_res_connect = self.stride == 1 and Cin == Cout
        self.upsample = self.stride == -1
        self.stride = abs(self.stride)
        groups = self.hidden_dim if g == 0 else g

        layers0 = [nn.UpsamplingNearest2d(scale_factor=2)] if self.upsample else []
        layers1 = [get_groupnorm(Cin, eps=BN_EPS),
                   ConvGNSwish(Cin, self.hidden_dim, k=1)]
        layers2 = [ConvGNSwish(self.hidden_dim, self.hidden_dim, stride=self.stride, groups=groups, k=k, dilation=dil),
                   Conv2D(self.hidden_dim, Cout, 1, 1, 0, bias=False, weight_norm=False),
                   get_groupnorm(Cout, eps=BN_EPS)]

        layers0.extend(layers1)
        self.conv1 = nn.Sequential(*layers0)
        self.conv2 = nn.Sequential(*layers2)

    def forward(self, x, temb=None):
        # ftr = self.conv1(x)
        ftr = checkpoint(self.conv1, x, preserve_rng_state=False)
        if temb is not None:
            ftr += temb
        # ftr = self.conv2(ftr)
        ftr = checkpoint(self.conv2, ftr, preserve_rng_state=False)
        return ftr


class Attention(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.gn_in = get_groupnorm(Cin)
        self.qkv = Conv2D(Cin, 3*Cin, kernel_size=1, stride=1, padding=0, weight_norm=True, bias=True)

    def forward(self, x):
        ftr = x
        hx, wx = x.size(2), x.size(3)

        # if feature map is bigger, average pool
        up_sample = False
        max_dim = 16
        if hx > max_dim:
            scale_factor = hx // max_dim   # 16 should be a divider of height
            ftr = F.avg_pool2d(ftr, kernel_size=scale_factor, stride=scale_factor)
            up_sample = True

        ftr = self.gn_in(ftr)
        qkv = self.qkv(ftr)

        # view in Linear has a bug that causes OOM:
        # https://github.com/pytorch/pytorch/pull/48696
        with autocast(False):
            # compute attention scores
            b, c, h, w = qkv.shape
            qkv = qkv.view(b, c, h * w)             # b, 3*c, hw
            q, k, v = torch.chunk(qkv, 3, dim=1)

            a = torch.bmm(k.permute(0, 2, 1), q)    # (b, hw, c) X (b, c, hw) -->  b, hw, hw
            a = a * ((c/3) ** (-0.5))
            a = F.softmax(a, dim=1)

            # use attention to compute values
            ftr = torch.bmm(v, a)  # (b, c, hw) X (b, hw', hw) -->  b, c, hw (aggregate softmax dim (hw')
            ftr = ftr.view(b, c//3, h, w)

        # ftr = self.projection(ftr)

        if up_sample:
            ftr = F.interpolate(ftr, scale_factor=scale_factor, mode='nearest')

        return ftr


class IdentityWithBackwardClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cutoff):
        ctx.cutoff = cutoff
        return x

    @staticmethod
    def backward(ctx, grad_output):
        cutoff = ctx.cutoff
        return grad_output.clamp_(min=-cutoff, max=cutoff), None


def identity_with_backward_clip(x, cutoff):
    return IdentityWithBackwardClip.apply(x, cutoff)


class IdentityWithBackwardClipNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cutoff):
        ctx.cutoff = cutoff
        return x

    @staticmethod
    def backward(ctx, grad_output):
        cutoff = ctx.cutoff
        hw = grad_output.shape[2] * grad_output.shape[3]
        norm_per_channel_normalized = torch.norm(grad_output, p=2, dim=[2, 3], keepdim=True) / hw
        c = cutoff / (norm_per_channel_normalized + 1e-6)
        c = torch.minimum(c, torch.ones_like(c))
        grad_output = grad_output * c
        return grad_output, None


def identity_with_backward_clip_norm(x, cutoff):
    return IdentityWithBackwardClipNorm.apply(x, cutoff)
