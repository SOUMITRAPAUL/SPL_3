"""
Coordinate Attention (CA) — Hou et al., CVPR 2021
Referenced as [37] in the paper. Used inside the DCA branch of DMRB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Decomposes global pooling into two 1-D poolings along H and W
    so that long-range dependencies along one spatial direction can
    be captured while preserving location information.
    """
    def __init__(self, in_ch, out_ch, reduction=32):
        super().__init__()
        mid = max(8, in_ch // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # squeeze W
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # squeeze H

        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.act   = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, out_ch, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Pool along each axis
        x_h = self.pool_h(x)                          # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)      # [B, C, W, 1]

        # Concatenate and encode
        y   = torch.cat([x_h, x_w], dim=2)            # [B, C, H+W, 1]
        y   = self.act(self.bn1(self.conv1(y)))        # [B, mid, H+W, 1]

        # Split back
        y_h, y_w = y.split([H, W], dim=2)

        a_h = torch.sigmoid(self.conv_h(y_h))          # [B, C, H, 1]
        a_w = torch.sigmoid(self.conv_w(y_w)).permute(0, 1, 3, 2)  # [B, C, 1, W]

        return x * a_h * a_w
