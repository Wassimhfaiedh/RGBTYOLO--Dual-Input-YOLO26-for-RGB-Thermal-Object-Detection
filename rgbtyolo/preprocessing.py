from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor


class RGBTPreprocessor:
    def __init__(self, image_size: int, device) -> None:
        self.image_size = image_size
        self.device = torch.device(device)

    def from_paths(self, rgb_path, thermal_path):
        rgb_path = Path(rgb_path)
        thermal_path = Path(thermal_path)

        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        if not thermal_path.is_file():
            raise FileNotFoundError(thermal_path)

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        thermal = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)

        if rgb is None or thermal is None:
            raise RuntimeError("Could not read RGB-Thermal pair.")

        return self.from_arrays(rgb, thermal)

    def from_arrays(self, rgb, thermal):
        if thermal.ndim == 3:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        size = (self.image_size, self.image_size)

        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        thermal = cv2.resize(thermal, size, interpolation=cv2.INTER_LINEAR)
        thermal_3c = np.repeat(thermal[..., None], 3, axis=2)

        return (
            rgb,
            thermal,
            self._tensor(rgb),
            self._tensor(thermal_3c),
        )

    def _tensor(self, image: np.ndarray) -> Tensor:
        return (
            torch.from_numpy(image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .to(self.device)
        )
