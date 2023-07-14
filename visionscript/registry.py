import logging

import supervision as sv


def yolov8_base(self):
    from ultralytics import YOLO

    # model name should be only letters and - and numbers

    model_name = self.state.get("current_active_model", "yolov8n")

    model_name = "".join(
        [
            letter
            for letter in model_name
            if letter.isalpha() or letter == "-" or letter.isdigit()
        ]
    )

    if self.state.get("model") and self.state["current_active_model"].lower() == "yolo":
        model = model
    else:
        model = YOLO(model_name + ".pt")

    inference_results = model(self.state["image_stack"][-1])[0]
    classes = inference_results.names

    logging.disable(logging.NOTSET)

    # Inference
    results = sv.Detections.from_yolov8(inference_results)

    return results, classes


def grounding_dino_base(self, classes):
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO

    mapped_items = {item: item for item in classes}

    base_model = GroundingDINO(CaptionOntology(mapped_items))

    inference_results = base_model.predict(self.state["last_loaded_image_name"])

    return inference_results, classes
