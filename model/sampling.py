"""
Sampling Blocks Based on Dual-Channel Superposition  (Section III-B)

Downsampling:
  Two parallel paths (different depth) both with stride-2 conv;
  concatenated → merged.  Different convolutions + activations improve
  details and non-linearity. Stride-2 reduces spatial size and increases
  feature channels.

Upsampling:
  Multiple convolutions for richer information + channel expression.
  Transposed convolution reduces channels (controls model complexity /
  computational cost).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSampleBlock(nn.Module):
    """
    Dual-channel superposition downsampling.
    Two branches with different depths are run in parallel (both stride 2),
    concatenated, then merged by a 1×1 conv.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        half = out_ch // 2   # each branch outputs half the channels

        # Branch 1 – shallow (single stride-2 conv)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, half, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Branch 2 – deeper (stride-2 conv followed by a second conv)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, half, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(half, half, 3, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Merge concatenated channels → out_ch
        self.merge = nn.Conv2d(out_ch, out_ch, 1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return self.merge(torch.cat([b1, b2], dim=1))


class UpSampleBlock(nn.Module):
    """
    Upsampling block.
    Multiple convolutions enrich the feature representation;
    transposed convolution reduces channels while doubling spatial size.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.enrich = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Transposed conv: reduces channels and doubles spatial size
        self.tconv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.enrich(x)
        x = self.act(self.tconv(x))
        return x
