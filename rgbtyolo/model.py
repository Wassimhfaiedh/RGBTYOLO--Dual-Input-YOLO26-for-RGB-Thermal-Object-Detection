from pathlib import Path

import torch

from .predictor import RGBTPredictor


class RGBTYOLO:
    """High-level RGB-Thermal inference API."""

    def __init__(self, weights, *, device=None):
        self.weights = Path(weights)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)

        checkpoint = torch.load(
            self.weights,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
            self.checkpoint = {}
        elif isinstance(checkpoint, dict) and isinstance(
            checkpoint.get("model"),
            torch.nn.Module,
        ):
            self.model = checkpoint["model"]
            self.checkpoint = checkpoint
        else:
            raise RuntimeError(
                "Inference API expects a self-contained checkpoint. "
                "Use load_state_dict_checkpoint() with DualInputYOLO26 "
                "for training state_dict checkpoints."
            )

        self.image_size = int(self.checkpoint.get("image_size", 640))
        self.num_classes = int(
            self.checkpoint.get("number_of_classes", 2)
        )

        self.model = self.model.to(self.device).eval()
        self.predictor = RGBTPredictor(
            self.model,
            device=self.device,
            image_size=self.image_size,
            num_classes=self.num_classes,
        )

    def predict(self, *, rgb, thermal, conf=0.25):
        return self.predictor.predict(
            rgb=rgb,
            thermal=thermal,
            conf=conf,
        )

    def predict_video(
        self,
        *,
        rgb_video,
        thermal_video,
        output="runs/result.mp4",
        conf=0.25,
        display=False,
    ):
        return self.predictor.predict_video(
            rgb_video=rgb_video,
            thermal_video=thermal_video,
            output=output,
            conf=conf,
            display=display,
        )
