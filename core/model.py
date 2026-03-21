"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGainPerDomain(nn.Module):
    def __init__(self, num_domains: int, init_gain=1.0, eps=1e-6):
        super().__init__()
        self.num_domains = num_domains
        self.eps = eps
        raw_init = torch.log(torch.expm1(torch.tensor(init_gain, dtype=torch.float32)))
        self.raw_weight = nn.Parameter(raw_init.expand(num_domains, 4).clone())  # [D,4]

    def _get_gain(self, d_idx: torch.Tensor):
        g = F.softplus(self.raw_weight[d_idx]) + self.eps # [B,4]
        return g.view(-1, 4, 1, 1)

    def forward(self, x, domain):
        return x * self._get_gain(domain)

    def inverse(self, x, domain):
        return x / self._get_gain(domain)

class SpatialKernelGateCond(nn.Module):
    def __init__(self, channels: int, style_dim: int, k = 7, dilations=None):
        super().__init__()
        assert k % 2 == 1, "k must be odd to keep size"
        pad = k // 2

        self.multi = dilations is not None
        if not self.multi:
            self.dw = nn.Conv2d(channels, channels, k, padding=pad,
                                groups=channels, bias=False)
            self.fuse = nn.Identity()
        else:
            self.branches = nn.ModuleList([
                nn.Conv2d(channels, channels, k,
                          padding=d*(k//2), dilation=d,
                          groups=channels, bias=False)
                for d in dilations
            ])
            self.fuse = nn.Conv2d(channels*len(dilations), channels, 1, bias=True)

        self.style_fc = nn.Sequential(
            nn.Linear(style_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.groups == channels:
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        nn.init.zeros_(self.style_fc[-1].weight)
        nn.init.zeros_(self.style_fc[-1].bias)

        self.att_scale = 1.0

    def forward(self, x, s):
        if self.multi:
            h = torch.cat([b(x) for b in self.branches], dim=1)
            att_sp = torch.sigmoid(self.fuse(h))          # [B,C,H,W]
        else:
            att_sp = torch.sigmoid(self.dw(x))            # [B,C,H,W]

        att_ch = torch.sigmoid(self.style_fc(s)).view(s.size(0), -1, 1, 1)  # [B,C,1,1]
        att = att_sp * att_ch
        return x * (1.0 + self.att_scale * att)

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64,
                 actv=nn.LeakyReLU(0.2), upsample=False, use_spa_gate = True, spa_k=7, spa_multi=True):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

        self.use_spa_gate = use_spa_gate
        if use_spa_gate:
            if spa_multi:
                self.spa_gate = SpatialKernelGateCond(dim_out, style_dim, k=spa_k, dilations=(1,4,9))
            else:
                self.spa_gate = SpatialKernelGateCond(dim_out, style_dim, k=spa_k, dilations=None)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.conv1(x)

        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)

        if self.use_spa_gate:
            x = self.spa_gate(x, s)

        return x

    def forward(self, x, s):
        res = self._residual(x, s)
        skip = self._shortcut(x)
        return (res + skip) / math.sqrt(2)

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, num_domains=2, use_chan_gain=True, init_gain=1.0, eps=1e-6):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.use_chan_gain = use_chan_gain
        if use_chan_gain:
            self.chan_gain = ChannelGainPerDomain(num_domains=num_domains, init_gain=init_gain, eps=eps)

        self.from_raw = nn.Conv2d(4, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_raw = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 4, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x, s, y_org=None, c_t=None):
        if self.use_chan_gain and (y_org is not None):
            x = self.chan_gain.inverse(x, y_org)

        x = self.from_raw(x)
        skips = []
        for block in self.encode:
            skips.append(x)
            x = block(x)

        for idx, block in enumerate(self.decode):
            x = block(x, s)
            skip = skips[-idx-1]
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bicubic', align_corners=False)
            x = x + skip
        x = self.to_raw(x)

        if self.use_chan_gain and (c_t is not None):
            x = self.chan_gain(x, c_t)

        return x

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(4, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(4, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, num_domains=args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    generator_ema = copy.deepcopy(generator)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema
