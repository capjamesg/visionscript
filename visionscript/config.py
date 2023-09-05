import os

import numpy as np
import psutil
import supervision as sv
import torch
from PIL import Image

from visionscript import registry
from visionscript.pose import Pose

DATA_TYPES = {
    sv.Detections: "Detection",
    np.ndarray: "Image",
    torch.Tensor: "Image",
    Image.Image: "Image",
    str: "String",
    int: "Integer",
    Pose: "Pose",
}

STACK_MAXIMUM = {
    "image_stack": {
        # 50% of available memory
        "maximum": 0.5 * psutil.virtual_memory().available,
        "also_reset": ["detections_stack"],
    }
}

CONCURRENT_MAXIMUM = 10

VIDEO_STRIDE = 2

CACHE_DIRECTORY = os.path.join(os.path.expanduser("~"), ".visionscript")

FASTSAM_DIR = os.path.join(CACHE_DIRECTORY, "FastSAM")
FASTSAM_WEIGHTS_DIR = os.path.join(FASTSAM_DIR, "weights")

CONCURRENT_VIDEO_TRANSFORMATIONS = ["showtext", "greyscale", "show"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FILE_SIZE = 10000000  # 10MB

SUPPORTED_INFERENCE_MODELS = {
    "groundingdino": lambda self, classes: registry.grounding_dino_base(self, classes),
    "yolov8": lambda self, classes: registry.yolov8_base(self, classes),
    "fastsam": lambda self, classes: registry.fast_sam_base(self, classes),
    "yolov8s-pose": lambda self, _: registry.yolov8_pose_base(self, _),
    "roboflow": lambda self, _: registry.use_roboflow_hosted_inference(self, _),
}

SUPPORTED_TRAIN_MODELS = {
    "vit": lambda self, folder: registry.vit_target(self, folder),
    "yolov8": lambda self, folder: registry.yolov8_target(self, folder),
}

ALIASED_FUNCTIONS = {
    "isita": "classify",
    "find": "detect",
    "describe": "caption",
    "getcolors": "getcolours",
}
