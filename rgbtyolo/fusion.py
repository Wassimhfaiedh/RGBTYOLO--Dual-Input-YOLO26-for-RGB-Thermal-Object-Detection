from __future__ import annotations

import torch
from torch import Tensor, nn


class P5ConcatFusion(nn.Module):
    """P5 RGB-Thermal channel fusion."""

    def __init__(
        self,
        rgb_channels: int = 256,
        thermal_channels: int = 256,
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                rgb_channels + thermal_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb_p5: Tensor, thermal_p5: Tensor) -> Tensor:
        if rgb_p5.shape[-2:] != thermal_p5.shape[-2:]:
            raise ValueError("RGB and thermal P5 spatial sizes must match.")
        return self.block(torch.cat((rgb_p5, thermal_p5), dim=1))
