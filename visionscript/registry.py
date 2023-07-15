import supervision as sv
import logging
import numpy as np
import torch
import sys
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def yolov8_base(self, _) -> sv.Detections:
    from ultralytics import YOLO

    if self.state.get("model") and self.state["current_active_model"].lower() == "yolo":
        model = model
    else:
        model = YOLO("yolov8n.pt")

    inference_results = model(self.state["image_stack"][-1])[0]
    classes = inference_results.names

    logging.disable(logging.NOTSET)

    # Inference
    results = sv.Detections.from_yolov8(inference_results)

    return results, classes


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
    import os

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

    # text prompt
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
        confidence=np.array([1]),
    )

    return detections


def yolov8_target(self, folder):
    if "autodistill_yolov8" not in sys.modules:
        from autodistill_yolov8 import YOLOv8

    base_model = YOLOv8("yolov8n.pt")

    model = base_model.train(os.path.join(folder, "data.yaml"), epochs=10)

    return model


def vit_target(self, folder):
    if "autodistill_vit" not in sys.modules:
        import autodistill_vit as ViT

    base_model = ViT("ViT-B/32")

    model = base_model.train(folder, "ViT-B/32")

    return model
