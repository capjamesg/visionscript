import contextlib
import io
import json
import logging
import os
import sys

import numpy as np
import supervision as sv
import torch
from PIL import Image
from roboflow import Roboflow

from visionscript.error_handling import ModelNotAvailable
from visionscript.rf_models import STANDARD_ROBOFLOW_MODELS

if os.environ.get("ROBOFLOW_API_KEY"):
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
else:
    rf = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "visionscript")

# retrieve rf_models.json from ~/.cache/visionscript
# this is where the user keeps a registry of custom models
# which is combined with the standard RF models
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

if not os.path.exists(os.path.join(CACHE_DIR, "rf_models.json")):
    with open(os.path.join(CACHE_DIR, "rf_models.json"), "w") as f:
        json.dump({}, f)

with open(os.path.join(CACHE_DIR, "rf_models.json"), "r") as f:
    ROBOFLOW_MODELS = json.load(f)

ROBOFLOW_MODELS = {**ROBOFLOW_MODELS, **STANDARD_ROBOFLOW_MODELS}


def use_roboflow_hosted_inference(self, _) -> list:
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if not rf:
            raise ModelNotAvailable("The model you have tried to use requires an API key. Run export ROBOFLOW_API_KEY='' with your API key and run your script again.")

        # base64 image

        image = self._get_item(-1, "image_stack")
        # PIL to base64
        buffered = io.BytesIO()

        # read into PIL

        # bgr to rgb
        image = image[:, :, ::-1]
        image = Image.fromarray(image)

        if self.state.get("last_loaded_image_name") and self.state.get(
            "last_loaded_image_name", ""
        ).endswith(".jpg"):
            image.save(buffered, format="JPEG")
        else:
            image.save(buffered, format="PNG")

        if self.state.get("model") is None:
            model = ROBOFLOW_MODELS.get(
                self.state["current_active_model"].lower().split("roboflow")[1].strip()
            )
            model["labels"] = [l.lower() for l in model["labels"]]
            project = rf.workspace().project(model["model_id"])

            self.state["last_classes"] = [
                i.lower() for i in list(sorted(project.classes.keys()))
            ]

            if os.environ.get("ROBOFLOW_INFER_SERVER_DESTINATION"):
                inference_model = project.version(
                    model["version"],
                    local=os.environ.get("ROBOFLOW_INFER_SERVER_DESTINATION"),
                ).model
            else:
                inference_model = project.version(model["version"]).model

            self.state["model"] = inference_model
        else:
            model = ROBOFLOW_MODELS.get(
                self.state["current_active_model"].lower().split("roboflow")[1].strip()
            )
            model["labels"] = [l.lower() for l in model["labels"]]
            inference_model = self.state["model"]

        with open("temp.jpg", "wb") as f:
            f.write(buffered.getvalue())

        predictions = inference_model.predict("temp.jpg", confidence=0.3)
        predictions = predictions.json()

        for p in predictions["predictions"]:
            p["class"] = p["class"].lower()

        classes = [i.lower() for i in list(sorted(self.state["last_classes"]))]
        not_sorted_classes = [i.lower() for i in self.state["last_classes"]]

        processed_detections = sv.Detections.from_roboflow(predictions, classes)

        idx_to_class = {idx: item for idx, item in enumerate(not_sorted_classes)}

        return processed_detections, idx_to_class, ",".join(model["labels"])


def yolov8_pose_base(self, _) -> list:
    # returns 1x17 vector
    from ultralytics import YOLO

    logging.disable(logging.CRITICAL)

    if self.state.get("model") and self.state["current_active_model"].lower() == "yolo":
        model = model
    else:
        model = YOLO("yolov8s-pose.pt")

    inference_results = model(self._get_item(-1, "image_stack"))[0]

    logging.disable(logging.NOTSET)

    return inference_results.keypoints[0]


def yolov8_base(self, user_classes) -> sv.Detections:
    from ultralytics import YOLO

    if self.state.get("model") and self.state["current_active_model"].lower() == "yolo":
        model = model
    else:
        model = YOLO("yolov8n.pt")

    inference_results = model(self._get_item(-1, "image_stack"))[0]
    classes = inference_results.names

    logging.disable(logging.NOTSET)

    results = sv.Detections.from_yolov8(inference_results)

    return results, classes, user_classes


def grounding_dino_base(self, classes) -> sv.Detections:
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO

    mapped_items = {item: item for item in classes}

    base_model = GroundingDINO(CaptionOntology(mapped_items))

    inference_results = base_model.predict(self.state["last_loaded_image_name"])

    return inference_results, classes


def fast_sam_base(self, text_prompt) -> sv.Detections:
    from autodistill_fastsam import FastSAM
    from autodistill.detection import CaptionOntology

    mapped_items = {item: item for item in text_prompt}

    base_model = FastSAM(CaptionOntology(mapped_items))

    inference_results = base_model.predict(self.state["last_loaded_image_name"])

    return inference_results, text_prompt


def yolov8_target(self, folder):
    # if "autodistill_yolov8" not in sys.modules:
    #     from autodistill_yolov8 import YOLOv8

    base_model = YOLOv8("yolov8n.pt")

    model = base_model.train(os.path.join(folder, "data.yaml"), epochs=10)

    return model, model.names


def vit_target(self, folder):
    if "autodistill_vit" not in sys.modules:
        import autodistill_vit as ViT

    base_model = ViT("ViT-B/32")

    model = base_model.train(folder, "ViT-B/32")

    return model, model.names
