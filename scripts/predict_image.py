import argparse
from rgbtyolo import RGBTYOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--rgb", required=True)
parser.add_argument("--thermal", required=True)
parser.add_argument("--output", default="runs/result.jpg")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--device", default=None)
args = parser.parse_args()

model = RGBTYOLO(args.weights, device=args.device)
result = model.predict(
    rgb=args.rgb,
    thermal=args.thermal,
    conf=args.conf,
)
result.save(args.output)
print(result)
