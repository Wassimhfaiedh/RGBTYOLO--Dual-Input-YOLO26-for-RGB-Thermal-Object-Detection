from pathlib import Path

IMAGE_SIZE = 640
NUM_CLASSES = 2
CLASS_NAMES = {0: "person", 1: "car"}
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_VIDEO_OUTPUT = Path("runs/predict_video/result.mp4")
