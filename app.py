from rgbtyolo import RGBTYOLO

model = RGBTYOLO(
    "weights/rgbtyolo_v1.pt",
    device="cpu",
)

model.predict_video(
    rgb_video="videos/rgb.mp4",
    thermal_video="videos/thermal.mp4",
    output="runs/result.mp4",
    conf=0.25,
    display=True,
)
