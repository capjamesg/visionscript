import contextlib
import io
import logging
import os
import sys
import tempfile

import numpy as np
import supervision as sv
import torch
from PIL import Image
from roboflow import Roboflow
from visionscript.rf_models import ROBOFLOW_MODELS

if os.environ.get("ROBOFLOW_API_KEY"):
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
else:
    rf = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def use_roboflow_hosted_inference(self, _) -> list:
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if not rf:
            return sv.Detections.empty(), [], ""

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
            project = rf.workspace().project(model["model_id"])

            self.state["last_classes"] = list(sorted(project.classes.keys()))

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
            inference_model = self.state["model"]

        with open("temp.jpg", "wb") as f:
            f.write(buffered.getvalue())

        predictions = inference_model.predict("temp.jpg", confidence=0.3)

        processed_detections = sv.Detections.from_roboflow(
            predictions.json(), list(sorted(self.state["last_classes"]))
        )

        idx_to_class = {idx: item for idx, item in enumerate(self.state["last_classes"])}

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


def yolov8_base(self, _) -> sv.Detections:
    from ultralytics import YOLO

    if self.state.get("model") and self.state["current_active_model"].lower() == "yolo":
        model = model
    else:
        model = YOLO("yolov8n.pt")

    inference_results = model(self._get_item(-1, "image_stack"))[0]
    classes = inference_results.names

    logging.disable(logging.NOTSET)

    results = sv.Detections.from_yolov8(inference_results)

    return results, classes, ",".join(classes)


def grounding_dino_base(self, classes) -> sv.Detections:
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO

    mapped_items = {item: item for item in classes}

    base_model = GroundingDINO(CaptionOntology(mapped_items))

    inference_results = base_model.predict(self.state["last_loaded_image_name"])

    return inference_results, classes


def fast_sam_base(self, text_prompt) -> sv.Detections:
    from .FastSAM.fastsam import FastSAM, FastSAMPrompt

    logging.disable(logging.CRITICAL)
    # get current path

    current_path = os.getcwd()

    model = FastSAM(os.path.join(current_path, "weights", "FastSAM.pt"))

    everything_results = model(
        self.state["last_loaded_image_name"],
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt_process = FastSAMPrompt(
        self.state["last_loaded_image_name"], everything_results, device=DEVICE
    )

    if "," in text_prompt:
        ann = []

        text_prompt = text_prompt.split(",")

        for prompt in text_prompt:
            ann.extend(prompt_process.text_prompt(text=prompt))
    else:
        ann = prompt_process.text_prompt(text=text_prompt)

    logging.disable(logging.NOTSET)

    results = []
    class_ids = []

    for mask in ann:
        results.append(
            sv.Detections(
                mask=np.array([mask]),
                xyxy=sv.mask_to_xyxy(np.array([mask])),
                class_id=np.array([0]),
                confidence=np.array([1]),
            )
        )
        class_ids.append(0)

    detections = sv.Detections(
        mask=np.array([item.mask[0] for item in results]),
        xyxy=np.array([item.xyxy[0] for item in results]),
        class_id=np.array(class_ids),
        confidence=np.array([1] * len(results)),
    )

    return detections, ["text"], "text"


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
