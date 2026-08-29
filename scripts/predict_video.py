import argparse
from rgbtyolo import RGBTYOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--rgb", required=True)
parser.add_argument("--thermal", required=True)
parser.add_argument("--output", default="runs/result.mp4")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--device", default=None)
parser.add_argument("--display", action="store_true")
args = parser.parse_args()

model = RGBTYOLO(args.weights, device=args.device)
path = model.predict_video(
    rgb_video=args.rgb,
    thermal_video=args.thermal,
    output=args.output,
    conf=args.conf,
    display=args.display,
)
print(f"Saved: {path}")
