from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import Tensor, nn

from .fusion import P5ConcatFusion


class DualInputYOLO26(nn.Module):
    """Dual-input YOLO26 with P5 cross-modal feature fusion."""

    BACKBONE_END = 10
    NECK_START = 11

    def __init__(
        self,
        rgb_detector: nn.Module,
        thermal_detector: nn.Module,
        *,
        p5_channels: int = 256,
    ) -> None:
        super().__init__()

        rgb_layers = list(rgb_detector.model)
        thermal_layers = list(thermal_detector.model)

        if len(rgb_layers) != len(thermal_layers):
            raise ValueError("RGB and thermal YOLO26 graphs must match.")
        if len(rgb_layers) <= self.NECK_START:
            raise ValueError("Unexpected YOLO26 graph.")

        self.rgb_backbone = nn.ModuleList(
            deepcopy(rgb_layers[: self.NECK_START])
        )
        self.thermal_backbone = nn.ModuleList(
            deepcopy(thermal_layers[: self.NECK_START])
        )
        self.fusion_p5 = P5ConcatFusion(
            p5_channels,
            p5_channels,
            p5_channels,
        )
        self.neck_head = nn.ModuleList(
            deepcopy(rgb_layers[self.NECK_START :])
        )

        for name in ("names", "nc", "stride", "yaml"):
            if hasattr(rgb_detector, name):
                setattr(self, name, deepcopy(getattr(rgb_detector, name)))

    @classmethod
    def from_ultralytics(
        cls,
        rgb_weights: str,
        thermal_weights: str,
        *,
        p5_channels: int = 256,
    ) -> "DualInputYOLO26":
        from ultralytics import YOLO

        return cls(
            YOLO(rgb_weights).model,
            YOLO(thermal_weights).model,
            p5_channels=p5_channels,
        )

    def forward(self, rgb: Tensor, thermal: Tensor) -> Any:
        rgb_cache = self._run_layers(rgb, self.rgb_backbone, 0, [])
        thermal_cache = self._run_layers(
            thermal,
            self.thermal_backbone,
            0,
            [],
        )

        fused_p5 = self.fusion_p5(
            rgb_cache[-1],
            thermal_cache[-1],
        )

        cache = list(rgb_cache)
        cache[-1] = fused_p5
        x = fused_p5

        for local_index, module in enumerate(self.neck_head):
            global_index = self.NECK_START + local_index
            x = self._select_input(module, x, cache, global_index)
            x = module(x)
            cache.append(x)

        return x

    def freeze_backbones(self) -> None:
        self._set_trainable(self.rgb_backbone, False)
        self._set_trainable(self.thermal_backbone, False)
        self._set_trainable(self.fusion_p5, True)
        self._set_trainable(self.neck_head, True)

    def unfreeze_all(self) -> None:
        self._set_trainable(self, True)

    @classmethod
    def _run_layers(
        cls,
        x: Tensor,
        layers: nn.ModuleList,
        offset: int,
        cache: list,
    ) -> list:
        for local_index, module in enumerate(layers):
            index = offset + local_index
            x = cls._select_input(module, x, cache, index)
            x = module(x)
            cache.append(x)
        return cache

    @staticmethod
    def _select_input(module, x, cache, current_index):
        source = getattr(module, "f", -1)

        if source == -1:
            return x

        if isinstance(source, int):
            index = source if source >= 0 else current_index + source
            return cache[index]

        values = []
        for item in source:
            if item == -1:
                values.append(x)
            else:
                index = item if item >= 0 else current_index + item
                values.append(cache[index])
        return values

    @staticmethod
    def _set_trainable(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = value
