from pathlib import Path

import torch


def load_state_dict_checkpoint(
    path,
    model,
    *,
    device="cpu",
    strict=True,
):
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def export_inference_checkpoint(
    path,
    model,
    *,
    image_size=640,
    class_names=None,
    metrics=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.cpu().eval(),
        "architecture": "Dual-Input YOLO26 P5 Concat + 1x1 Conv",
        "image_size": image_size,
        "number_of_classes": 2,
        "class_names": class_names or {0: "person", 1: "car"},
        "metrics": metrics or {},
        "version": "1.0.0",
    }, path)

    return path
