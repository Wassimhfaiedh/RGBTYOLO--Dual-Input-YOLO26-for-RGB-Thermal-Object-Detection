from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import torch

from .constants import IMAGE_EXTENSIONS
from .preprocessing import RGBTPreprocessor
from .results import RGBTResults


class RGBTPredictor:
    def __init__(self, model, *, device, image_size=640, num_classes=2):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.preprocessor = RGBTPreprocessor(image_size, device)

    @property
    def image_size(self):
        return self.preprocessor.image_size

    def predict(self, *, rgb, thermal, conf=0.25):
        rgb_image, thermal_image, rgb_tensor, thermal_tensor = (
            self.preprocessor.from_paths(rgb, thermal)
        )
        detections, latency = self._infer(
            rgb_tensor, thermal_tensor, conf
        )
        return RGBTResults(
            rgb_image=rgb_image,
            thermal_image=thermal_image,
            detections=detections,
            rgb_path=rgb,
            thermal_path=thermal,
            inference_time_ms=latency,
        )

    def predict_arrays(self, rgb_image, thermal_image, *, conf=0.25):
        rgb, thermal, rgb_tensor, thermal_tensor = (
            self.preprocessor.from_arrays(rgb_image, thermal_image)
        )
        detections, latency = self._infer(
            rgb_tensor, thermal_tensor, conf
        )
        return RGBTResults(
            rgb_image=rgb,
            thermal_image=thermal,
            detections=detections,
            rgb_path="rgb_frame",
            thermal_path="thermal_frame",
            inference_time_ms=latency,
        )

    def predict_video(
        self,
        *,
        rgb_video,
        thermal_video,
        output,
        conf=0.25,
        display=False,
    ):
        rgb_cap = cv2.VideoCapture(str(rgb_video))
        thermal_cap = cv2.VideoCapture(str(thermal_video))

        if not rgb_cap.isOpened() or not thermal_cap.isOpened():
            raise RuntimeError("Could not open RGB-Thermal videos.")

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = rgb_cap.get(cv2.CAP_PROP_FPS) or 25.0

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (self.image_size * 2, self.image_size),
        )

        try:
            while True:
                ok_rgb, rgb = rgb_cap.read()
                ok_th, thermal = thermal_cap.read()

                if not ok_rgb or not ok_th:
                    break

                result = self.predict_arrays(
                    rgb,
                    thermal,
                    conf=conf,
                )
                frame = cv2.cvtColor(
                    result.plot(),
                    cv2.COLOR_RGB2BGR,
                )
                writer.write(frame)

                if display:
                    cv2.imshow("RGBTYOLO Video", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            rgb_cap.release()
            thermal_cap.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()

        return output

    def _infer(self, rgb, thermal, conf):
        start = perf_counter()
        with torch.inference_mode():
            output = self.model(rgb, thermal)
        latency = (perf_counter() - start) * 1000.0

        detections = output[0]
        if detections.ndim == 3:
            detections = detections[0]

        valid = (
            torch.isfinite(detections).all(dim=1)
            & (detections[:, 4] >= conf)
            & (detections[:, 5] >= 0)
            & (detections[:, 5] < self.num_classes)
            & (detections[:, 2] > detections[:, 0])
            & (detections[:, 3] > detections[:, 1])
        )

        return detections[valid].detach().cpu().float(), latency
