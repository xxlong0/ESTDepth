from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

class EpipolarTransformer(nn.Module):
    # combine GRU and transformer
    def __init__(self, input_channel, output_channel, kernel_size):
        super(EpipolarTransformer, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.gamma = torch.nn.Parameter(torch.zeros(1))

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv3d(gru_input_channel, output_channel * 2, kernel_size, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv3d(gru_input_channel, output_channel, kernel_size, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x D x H x W
        # h = N x C x D x H x W

        # c = N x C*2 x D x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x D x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = torch.nn.functional.sigmoid(rn)
        uns = torch.nn.functional.sigmoid(un)
        return rns, uns

    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, target_key, target_value, warped_values=None, warped_keys=None):
        """
        return the fused volume of target_volume
        """
        B, C, D, H, W = target_value.shape

        if warped_values is not None:
            correlations = []
            for key in warped_keys:
                correlation = torch.sum(target_key * key, dim=1, keepdim=True)  # [B,1,D,H,W]
                correlations.append(correlation)

            correlations = torch.stack(correlations, dim=-1)  # [B,1,D,H,W, N]
            attention_maps = self.softmax(correlations)  # [B,1,D,H,W, N]

            values = torch.stack(warped_values, dim=-1)  # [B,C,D,H,W, N]

            h = torch.mean(values * attention_maps.repeat(1, C, 1, 1, 1, 1), dim=-1, keepdim=False)
        else:
            h = None

        HC = self.output_channel
        if (h is None):
            h = torch.zeros((B, HC, D, H, W), dtype=torch.float, device=target_value.device)
        r, u = self.gates(target_value, h)
        o = self.output(target_value, h, r, u)
        y = self.activation(o)
        return u * h + (1 - u) * y


