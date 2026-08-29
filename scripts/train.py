import torch

from rgbtyolo import DualInputYOLO26
from rgbtyolo.training import TrainingStage, fit_stage


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PHASE1 = TrainingStage(
    epochs=30,
    learning_rate=1e-4,
    patience=10,
)

PHASE2 = TrainingStage(
    epochs=30,
    learning_rate=1e-5,
    patience=10,
)


def build_model():
    return DualInputYOLO26.from_ultralytics(
        "weights/yolo26_rgb.pt",
        "weights/yolo26_thermal.pt",
    ).to(DEVICE)


def train(train_loader, val_loader, criterion):
    model = build_model()

    model.freeze_backbones()
    fit_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device=DEVICE,
        stage=PHASE1,
        checkpoint_path="runs/train/phase1_best.pt",
    )

    model.unfreeze_all()
    fit_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device=DEVICE,
        stage=PHASE2,
        checkpoint_path="runs/train/phase2_best.pt",
    )

    return model
