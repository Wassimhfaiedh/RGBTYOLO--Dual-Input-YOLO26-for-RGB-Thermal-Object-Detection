from __future__ import annotations

from typing import Any

from torch import Tensor, nn


class YOLO26DetectionLoss(nn.Module):
    """Adapter for the native YOLO26 detection criterion."""

    def __init__(self, reference_detector: nn.Module) -> None:
        super().__init__()
        if not hasattr(reference_detector, "init_criterion"):
            raise TypeError("Detector does not expose init_criterion().")
        self.criterion = reference_detector.init_criterion()

    def forward(
        self,
        predictions: Any,
        batch: dict[str, Tensor],
    ):
        return self.criterion(predictions, batch)
