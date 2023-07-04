# VisualScript

# supress warnings
import warnings

warnings.filterwarnings("ignore")

import logging
import optparse

import clip
import cv2
import numpy as np
import supervision as sv
import torch
from lark import Lark
from PIL import Image
from ultralytics import YOLO

# state will have "last"
state = {
    "last": None,
    "last_function_type": None,
    "last_function_args": None,
    "last_loaded_image": None,
}


opt_parser = optparse.OptionParser()

opt_parser.add_option("--validate", action="store_true", dest="validate", default=False)
opt_parser.add_option("--ref", action="store_true", dest="ref", default=False)
opt_parser.add_option("--debug", action="store_true", dest="debug", default=False)
opt_parser.add_option("--file", action="store", dest="file", default=None)

options, args = opt_parser.parse_args()

USAGE = """
VisualScript (VIC) is a visual programming language for computer vision.

VisualScript is a line-based language. Each line is a function call.

Language Reference
------------------
Load["./abbey.jpg"] -> Load the image
Size[] -> Get the size of the image
Say[] -> Say the result of the last function
Detect["person"] -> Detect the person
Replace[] -> Replace the person with black image
Cutout[] -> Cutout the last detections
Count[] -> Count the last detections
CountInRegion[0, 0, 500, 500] -> Count the last detections in the region (x1, y1, x2, y2)
Classify["cat", "dog"] -> Classify the image in the provided categories
Save["./abbey2.jpg"] -> Save the last image

Example Program
---------------

Find a church in the image and cut it out.

Load["./abbey.jpg"]
Detect["church"]
Cutout[]
Save["./abbey2.jpg"]
"""

if options.ref:
    print(USAGE.strip())
    exit(0)

if options.debug:
    DEBUG = True
else:
    DEBUG = False

if options.file is not None:
    with open(options.file, "r") as f:
        code = f.read()
else:
    code = """
    Load["./abbey.jpg"]
    Size[]
    Say[]
    Detect["person"]
    Replace["person"]
    Save["./abbey2.jpg"]
    """

grammar = """
start: expr+

expr: classify | replace | load | save | say | detect | cutout | size | count | countinregion | EOL
classify: "Classify" "[" STRING ("," STRING)* "]"
replace: "Replace" "[" STRING "]"
load: "Load" "[" STRING "]"
save: "Save" "[" STRING "]"
say: "Say" "[" "]"
size: "Size" "[" "]"
cutout: "Cutout" "[" "]"
count: "Count" "[" "]"
countinregion: "CountInRegion" "[" INT "," INT "," INT "," INT "]"
detect: "Detect" "[" STRING ("," STRING)* "]"
EOL: "\\n"
INT: /-?\d+/
%import common.ESCAPED_STRING -> STRING
%import common.WS_INLINE
%ignore WS_INLINE
"""

parser = Lark(grammar)

try:
    tree = parser.parse(code.strip())
except Exception as e:
    print(e)
    exit(1)

if options.validate:
    print("Script is a valid VisualScript program.")
    exit(0)


def literal_eval(string):
    return string[1:-1]


def load(filename, _):
    return Image.open(filename)


def size(_, state):
    return state["last_loaded_image"].size


def cutout(_, state):
    x1, y1, x2, y2 = state["last"].xyxy[0]
    image = state["last_loaded_image"]
    state["last_loaded_image"] = image.crop((x1, y1, x2, y2))


def save(filename, state):
    state["last_loaded_image"].save(filename)


def count(args, state):
    if len(args) == 0:
        return len(state["last"].xyxy)
    else:
        return len([item for item in state["last"].class_id if item == args[0]])


def detect(classes, state):
    logging.disable(logging.CRITICAL)

    model = YOLO("yolov8n.pt")
    inference_results = model(state["last_loaded_image"])[0]

    logging.disable(logging.NOTSET)

    # Inference
    results = sv.Detections.from_yolov8(inference_results)

    inference_classes = inference_results.names

    classes = [key for key, item in inference_classes.items() if item in classes]

    results = results[np.isin(results.class_id, classes)]

    return results


def classify(labels, state):
    image = state["last"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # get idx of the most likely label class
        label_idx = probs.argmax()

        label_name = labels[label_idx]

    return label_name


def countInRegion(x1, y1, x2, y2, state):
    detections = state["last"]

    xyxy = detections.xyxy

    counter = 0

    for i in range(len(xyxy)):
        x1_, y1_, x2_, y2_ = xyxy[i]

        if x1_ >= x1 and y1_ >= y1 and x2_ <= x2 and y2_ <= y2:
            counter += 1

    return counter


def say(statement, state):
    if state.get("last_function_type", None) == "detect":
        last_args = state["last_function_args"]
        statement = "".join(
            [
                f"{last_args[i]} {state['last'].confidence[i]:.2f} {state['last'].xyxy[i]}\n"
                for i in range(len(state["last"].xyxy))
            ]
        )

    print(statement)


def replace(_, state):
    detections = state["last"]

    # random iamge is black xyxy
    xyxy = detections.xyxy

    random_img = np.zeros(
        (int(xyxy[0][3] - xyxy[0][1]), int(xyxy[0][2] - xyxy[0][0]), 3), np.uint8
    )

    # paste image
    state["last_loaded_image"].paste(
        Image.fromarray(random_img), (int(xyxy[0][0]), int(xyxy[0][1]))
    )


function_calls = {
    "load": lambda x, y: load(x, y),
    "save": lambda x, y: save(x, y),
    "classify": lambda x, y: classify(x, y),
    "size": lambda x, y: size(x, y),
    "say": lambda x, y: say(x, y),
    "detect": lambda x, y: detect(x, y),
    "cutout": lambda x, y: cutout(x, y),
    "count": lambda x, y: count(x, y),
    "countinregion": lambda x, y: countInRegion(*x, y),
    "replace": lambda x, y: replace(x, y),
}

if DEBUG:
    print(tree.children)

for node in tree.children:
    node = node.children[0]

    if not hasattr(node, "data"):
        continue

    # convert string to literal

    token = node.data

    func = function_calls[token.value]

    if token.value == "say":
        value = state["last"]
    else:
        # convert children to strings
        for item in node.children:
            if hasattr(item, "value"):
                if item.value.startswith('"') and item.value.endswith('"'):
                    item.value = literal_eval(item.value)
                else:
                    item.value = int(item.value)

        if len(node.children) == 1:
            value = node.children[0].value
        else:
            value = [item.value for item in node.children]

    result = func(value, state)

    state["last"] = result
    state["last_function_type"] = token.value
    state["last_function_args"] = [value]

    # if load
    if token.value == "load":
        state["last_loaded_image"] = result
