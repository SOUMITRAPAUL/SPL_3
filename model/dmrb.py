"""
DMRB – Deep Multi-Scale Residual Iterative Enhancement Block
Paper equation:  F_DMRB = DCA(F_IN) + W( IRB(F_CM) )

Components (from Fig. 2 and Section III-C):
  - F_CM   : two sets of grouped 3×3 convolutions applied to F_IN
  - IRB    : multi-scale residual cycle
               • parallel 3×3 and 5×5 group-convolutions  (local / global)
               • 1×1 conv to fuse
               • two short (skip) connections for cyclic residual iteration
  - W      : two 3×3 convolutions applied to IRB output
  - DCA    : two 3×3 conv layers + Coordinate Attention applied to F_IN
  - Output : DCA(F_IN)  ⊕  W(IRB(F_CM))
"""

import torch
import torch.nn as nn
from .attention import CoordinateAttention


# ─────────────────────────────────────────────
#  Residual Iteration Block (IRB)
# ─────────────────────────────────────────────
class IRB(nn.Module):
    """
    Multi-scale residual cycle processing.
    Uses 3×3 and 5×5 group-conv in parallel to extract local / global
    features; fuses with 1×1; two short connections provide cyclic residual.
    """
    def __init__(self, ch, num_groups=4):
        super().__init__()
        g = min(num_groups, ch)

        # parallel multi-scale extraction
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=g),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ch, ch, 5, padding=2, groups=g),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse  = nn.Conv2d(ch * 2, ch, 1)          # 1×1 fusion

        # second cycle (short connection 1)
        self.conv3b = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=g),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5b = nn.Sequential(
            nn.Conv2d(ch, ch, 5, padding=2, groups=g),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuseb  = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, x):
        # --- cycle 1 ---
        s3  = self.conv3(x)
        s5  = self.conv5(x)
        m   = self.fuse(torch.cat([s3, s5], dim=1))
        m   = m + x                          # short connection 1

        # --- cycle 2 ---
        s3b = self.conv3b(m)
        s5b = self.conv5b(m)
        mb  = self.fuseb(torch.cat([s3b, s5b], dim=1))
        out = mb + m                         # short connection 2
        return out


# ─────────────────────────────────────────────
#  DCA – Adaptive Deep Feature Processing
#  (two 3×3 convs  +  Coordinate Attention)
# ─────────────────────────────────────────────
class DCA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.ca = CoordinateAttention(ch, ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ca(out)
        return out


# ─────────────────────────────────────────────
#  DMRB
# ─────────────────────────────────────────────
class DMRB(nn.Module):
    """
    F_DMRB = DCA(F_IN)  +  W( IRB(F_CM) )

    F_CM is obtained by two sets of grouped 3×3 convolutions on F_IN.
    W    is two 3×3 convolutions on the IRB output.
    """
    def __init__(self, ch, num_groups=4):
        super().__init__()
        g = min(num_groups, ch)

        # F_CM: two grouped 3×3 convolutions
        self.fcm = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=g),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=g),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # IRB
        self.irb = IRB(ch, num_groups=g)

        # W: two 3×3 convolutions
        self.W = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

        # DCA branch on F_IN
        self.dca = DCA(ch)

    def forward(self, x):
        # F_DMRB = DCA(F_IN) + W(IRB(F_CM))
        f_cm   = self.fcm(x)
        irb_out = self.irb(f_cm)
        w_out  = self.W(irb_out)

        dca_out = self.dca(x)

        return dca_out + w_out
