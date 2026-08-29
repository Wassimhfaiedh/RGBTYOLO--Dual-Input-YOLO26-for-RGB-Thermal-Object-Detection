from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch import Tensor

from .constants import CLASS_COLORS, CLASS_NAMES


class RGBTResults:
    """Predictions for one synchronized RGB-Thermal pair."""

    def __init__(
        self,
        *,
        rgb_image: np.ndarray,
        thermal_image: np.ndarray,
        detections: Tensor,
        rgb_path,
        thermal_path,
        inference_time_ms: float,
    ) -> None:
        self.rgb_image = rgb_image
        self.thermal_image = thermal_image
        self.detections = detections
        self.rgb_path = Path(rgb_path)
        self.thermal_path = Path(thermal_path)
        self.inference_time_ms = inference_time_ms

    @property
    def boxes(self):
        return self.detections[:, :4]

    @property
    def scores(self):
        return self.detections[:, 4]

    @property
    def classes(self):
        return self.detections[:, 5].long()

    @property
    def count(self):
        return int(self.detections.shape[0])

    def summary(self):
        output = []
        for det in self.detections:
            x1, y1, x2, y2, score, class_id = det.tolist()
            class_id = int(class_id)
            output.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": float(score),
                "box": {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                },
            })
        return output

    def plot_rgb(self):
        return self._draw(self.rgb_image)

    def plot_thermal(self):
        image = cv2.cvtColor(self.thermal_image, cv2.COLOR_GRAY2RGB)
        return self._draw(image)

    def plot(self):
        return np.concatenate(
            (self.plot_rgb(), self.plot_thermal()),
            axis=1,
        )

    def show(self):
        cv2.imshow(
            "RGBTYOLO | RGB - Thermal",
            cv2.cvtColor(self.plot(), cv2.COLOR_RGB2BGR),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(self.plot(), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save {output_path}")
        return output_path

    def _draw(self, image):
        output = image.copy()
        h, w = output.shape[:2]

        for det in self.detections:
            x1, y1, x2, y2, score, class_id = det.tolist()
            class_id = int(class_id)

            x1 = int(np.clip(round(x1), 0, w - 1))
            y1 = int(np.clip(round(y1), 0, h - 1))
            x2 = int(np.clip(round(x2), 0, w - 1))
            y2 = int(np.clip(round(y2), 0, h - 1))

            color = CLASS_COLORS.get(class_id, (255, 255, 0))
            name = CLASS_NAMES.get(class_id, str(class_id))
            label = f"{name} {score:.2f}"

            cv2.rectangle(
                output, (x1, y1), (x2, y2),
                color, 2, cv2.LINE_AA,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )

            pad = 3
            label_h = th + baseline + 2 * pad
            top = max(0, y1 - label_h)
            bottom = y1 if y1 >= label_h else min(h - 1, y1 + label_h)
            right = min(w - 1, x1 + tw + 2 * pad)

            overlay = output.copy()
            cv2.rectangle(
                overlay, (x1, top), (right, bottom),
                color, -1,
            )
            cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)

            text_y = (
                y1 - baseline - pad
                if y1 >= label_h
                else min(h - 1, y1 + th + pad)
            )

            cv2.putText(
                output, label, (x1 + pad, text_y),
                font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA,
            )

        return output

    def __repr__(self):
        return (
            f"RGBTResults(detections={self.count}, "
            f"inference_time_ms={self.inference_time_ms:.2f})"
        )
