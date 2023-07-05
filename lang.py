import warnings

warnings.filterwarnings("ignore")

import logging
import mimetypes
import optparse
import os
import random
import string
import sys
import tempfile

import cv2
import numpy as np
import supervision as sv
from lark import Lark, UnexpectedCharacters, UnexpectedToken
from PIL import Image
from spellchecker import SpellChecker

from usage import (
    USAGE,
    language_grammar_reference,
    lowercase_language_grammar_reference,
)
from grammar import grammar

state = {
    "last": None,
    "last_function_type": None,
    "last_function_args": None,
    "last_loaded_image": None,
    "history": [],
}

spell = SpellChecker()

opt_parser = optparse.OptionParser()

opt_parser.add_option("--validate", action="store_true", dest="validate", default=False)
opt_parser.add_option("--ref", action="store_true", dest="ref", default=False)
opt_parser.add_option("--debug", action="store_true", dest="debug", default=False)
opt_parser.add_option("--file", action="store", dest="file", default=None)
opt_parser.add_option("--repl", action="store_true", dest="repl", default=False)

options, args = opt_parser.parse_args()

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

parser = Lark(grammar)


def handle_unexpected_characters(e):
    # raise error if class doesn't exist
    line = e.line
    column = e.column

    # check if function name in grammar
    function_name = code.strip().split("\n")[line - 1].split("[")[0].strip()

    language_grammar_reference_keys = language_grammar_reference.keys()

    if function_name in language_grammar_reference_keys:
        print(f"Syntax error on line {line}, column {column}.")
        print(f"Unexpected character: {e.char!r}")
        exit(1)

    spell.known(lowercase_language_grammar_reference)
    spell.word_frequency.load_words(lowercase_language_grammar_reference)

    alternatives = spell.candidates(function_name)

    if len(alternatives) == 0:
        print(f"Function {function_name} does not exist.")
        exit(1)

    print(f"Function '{function_name}' does not exist. Did you mean one of these?")
    print("-" * 10)

    for item in list(alternatives):
        if item in lowercase_language_grammar_reference:
            print(
                list(language_grammar_reference.keys())[
                    lowercase_language_grammar_reference.index(item.lower())
                ]
            )

    exit(1)


def handle_unexpected_token(e):
    line = e.line
    column = e.column

    print(f"Syntax error on line {line}, column {column}.")
    print(f"Unexpected token: {e.token!r}")
    exit(1)


if options.validate:
    print("Script is a valid VisualScript program.")
    exit(0)


def literal_eval(string):
    return string[1:-1]


def load(filename):
    if "requests" not in sys.modules:
        import requests
    if "validators" not in sys.modules:
        import validators

    if validators.url(filename):
        response = requests.get(filename)
        file_extension = mimetypes.guess_extension(response.headers["content-type"])

        # if not image, error
        print(file_extension)
        if file_extension not in (".png", ".jpg", ".jpeg"):
            print(f"File {filename} does not represent a png, jpg, or jpeg image.")
            exit(1)

        # 10 random characters
        filename = (
            "".join(
                random.choice(string.ascii_letters + string.digits) for _ in range(10)
            )
            + file_extension
        )

        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(response.content)
            filename = f.name

    if state.get("ctx") and state["ctx"].get("in"):
        filename = state["ctx"]["active_file"]

    state["last_loaded_image_name"] = filename

    return Image.open(filename)


def size(_):
    return state["last_loaded_image"].size


def cutout(_):
    x1, y1, x2, y2 = state["last"].xyxy[0]
    image = state["last_loaded_image"]
    state["last_loaded_image"] = image.crop((x1, y1, x2, y2))


def save(filename):
    state["last_loaded_image"].save(filename)


def count(args):
    if len(args) == 0:
        return len(state["last"].xyxy)
    else:
        return len([item for item in state["last"].class_id if item == args[0]])


def detect(classes):
    logging.disable(logging.CRITICAL)

    if "ultralytics" not in sys.modules:
        from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    inference_results = model(state["last_loaded_image"])[0]

    logging.disable(logging.NOTSET)

    # Inference
    results = sv.Detections.from_yolov8(inference_results)

    inference_classes = inference_results.names

    if len(classes) == 0:
        classes = inference_classes

    classes = [key for key, item in inference_classes.items() if item in classes]

    results = results[np.isin(results.class_id, classes)]

    return results


def classify(labels):
    image = state["last"]

    if state.get("model") and state["model"].__class__.__name__ == "ViT":
        model = state["model"]

        results = model.predict(image).get_top_k(1)

        if len(results.class_id) == 0:
            return sv.Classifications.empty()

        return results.class_id[0]
    elif state.get("model") and state["model"].__class__.__name__ == "YOLOv8":
        model = state["model"]

        results = model.predict(image)

        return results

    if "clip" not in sys.modules:
        import clip
        import torch

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


def segment(text_prompt):
    if "FastSAM" not in sys.modules:
        from fastsam import FastSAM, FastSAMPrompt

    logging.disable(logging.CRITICAL)
    model = FastSAM("./weights/FastSAM.pt")

    DEVICE = "cpu"
    everything_results = model(
        state["last_loaded_image_name"],
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt_process = FastSAMPrompt(
        state["last_loaded_image_name"], everything_results, device=DEVICE
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

    return sv.Detections(
        mask=np.array([item.mask[0] for item in results]),
        xyxy=np.array([item.xyxy[0] for item in results]),
        class_id=np.array(class_ids),
        confidence=np.array([1]),
    )


def countInRegion(x1, y1, x2, y2):
    detections = state["last"]

    xyxy = detections.xyxy

    counter = 0

    for i in range(len(xyxy)):
        x1_, y1_, x2_, y2_ = xyxy[i]

        if x1_ >= x1 and y1_ >= y1 and x2_ <= x2 and y2_ <= y2:
            counter += 1

    return counter


def say(statement):
    if state.get("last_function_type", None) in ("detect", "segment"):
        last_args = state["last_function_args"]
        statement = "".join(
            [
                f"{last_args[i]} {state['last'].confidence[i]:.2f} {state['last'].xyxy[i]}\n"
                for i in range(len(state["last"].xyxy))
            ]
        )

    print(statement)


def replace(filename):
    detections = state["last"]

    xyxy = detections.xyxy

    if filename is not None:
        random_img = Image.open(filename)
        # resize image
        random_img = random_img.resize(
            (int(xyxy[0][2] - xyxy[0][0]), int(xyxy[0][3] - xyxy[0][1]))
        )
    else:
        random_img = np.zeros(
            (int(xyxy[0][3] - xyxy[0][1]), int(xyxy[0][2] - xyxy[0][0]), 3), np.uint8
        )
        random_img = Image.fromarray(random_img)

    # paste image
    state["last_loaded_image"].paste(random_img, (int(xyxy[0][0]), int(xyxy[0][1])))


def train(folder):
    # if Detect or Classify run, train
    if "Detect" in state["history"]:
        if "autodistill_yolov8" not in sys.modules:
            import autodistill_yolov8 as YOLOv8

        base_model = YOLOv8("yolov8n.pt")

        model = base_model.train(folder, "yolov8n.pt")

    elif "Classify" in state["history"]:
        if "autodistill_vit" not in sys.modules:
            import autodistill_vit as ViT

        base_model = ViT("ViT-B/32")

        model = base_model.train(folder, "ViT-B/32")
    else:
        print("No training needed.")
        return

    state["model"] = model


def show(_):
    if state.get("last_function_type", None) == "detect":
        annotator = sv.BoxAnnotator()
    elif state.get("last_function_type", None) == "segment":
        annotator = sv.BoxAnnotator()
    else:
        annotator = None

    if state.get("last_loaded_image_name") is None or not os.path.exists(
        state["last_loaded_image_name"]
    ):
        print("Image does not exist.")
        return

    if annotator:
        image = annotator.annotate(
            cv2.imread(state["last_loaded_image_name"]), state["last"]
        )
    else:
        image = cv2.imread(state["last_loaded_image_name"])

    sv.plot_image(image, (8, 8))


# if None, the logic is handled in the main parser
function_calls = {
    "load": lambda x: load(x),
    "save": lambda x: save(x),
    "classify": lambda x: classify(x),
    "size": lambda x: size(x),
    "say": lambda x: say(x),
    "detect": lambda x: detect(x),
    "segment": lambda x: segment(x),
    "cutout": lambda x: cutout(x),
    "count": lambda x: count(x),
    "countinregion": lambda x: countInRegion(*x),
    "replace": lambda x: replace(x),
    "in": lambda x: None,
    "if": lambda x: None,
    "var": lambda x: None,
    "comment": lambda x: None,
    "expr": lambda x: None,
    "show": lambda x: show(x),
    "exit": lambda x: exit(0),
    "help": lambda x: print(language_grammar_reference[x]),
    "train": lambda x: train(x),
}


def parse_tree(tree):
    if not hasattr(tree, "children"):
        if hasattr(tree, "value") and tree.value.isdigit():
            return int(tree.value)

    for node in tree.children:
        # ignore EOLs, etc.
        if node == "True":
            return True
        elif node == "False":
            return False
        elif (hasattr(node, "type") and node.type == "INT") or isinstance(node, int):
            return int(node.value)
        elif not hasattr(node, "children") or len(node.children) == 0:
            node = node
        elif state.get("ctx") and (state["ctx"].get("in") or state["ctx"].get("if")):
            node = node
        else:
            node = node.children[0]

        if not hasattr(node, "data"):
            continue

        token = node.data

        if token == "comment":
            continue

        if token == "expr":
            parse_tree(node)
            continue

        if token.type == "BOOL":
            return node.children[0].value == "True"

        if token.type == "EQUALITY":
            return parse_tree(node.children[0]) == parse_tree(node.children[1])

        if token == "var":
            state[node.children[0].children[0].value] = parse_tree(node.children[1])
            state["last"] = state[node.children[0].children[0].value]
            continue

        if token.value == "if":
            statement = parse_tree(node.children[0])

            if statement is not False:
                context = node.children[3:]

                state["ctx"] = {
                    "if": True,
                }

                for item in context:
                    parse_tree(item)

                del state["ctx"]
                continue
            else:
                continue

        if token.value == None:
            continue

        func = function_calls[token.value]

        state["history"].append(token.value)

        if token.value == "say":
            value = state["last"]
            func(value)
            continue
        else:
            # convert children to strings
            for item in node.children:
                if hasattr(item, "value"):
                    if item.value.startswith('"') and item.value.endswith('"'):
                        item.value = literal_eval(item.value)
                    elif item.type in ("EOL", "INDENT", "DEDENT"):
                        continue
                    elif item.type == "STRING":
                        item.value = literal_eval(item.value)
                    else:
                        item.value = int(item.value)

        if token.value == "in":
            state["ctx"] = {
                "in": os.listdir(node.children[0].value),
            }

            for file_name in state["ctx"]["in"]:
                state["ctx"]["active_file"] = os.path.join(
                    literal_eval(node.children[0]), file_name
                )
                # ignore first 2, then do rest
                context = node.children[3:]

                for item in context:
                    parse_tree(item)

            del state["ctx"]

            continue

        if len(node.children) == 1:
            value = node.children[0].value
        else:
            value = [item.value for item in node.children]

        result = func(value)

        state["last"] = result
        state["last_function_type"] = token.value
        state["last_function_args"] = [value]

        if token.value == "load":
            state["last_loaded_image"] = result


if options.repl:
    print("Welcome to VisualScript!")
    print("Type 'Exit[]' to exit.")
    print("Read the docs at https://visualscript.org/docs")
    print("For help, type 'Help[FunctionName]'.")
    print("-" * 20)
    while True:
        code = input(">>> ")

        try:
            tree = parser.parse(code.strip())
        except UnexpectedCharacters as e:
            handle_unexpected_characters(e)
        except UnexpectedToken as e:
            handle_unexpected_token(e)

        parse_tree(tree)

if __name__ == "__main__":
    if options.file is None:
        try:
            tree = parser.parse(code.strip())
        except UnexpectedCharacters as e:
            handle_unexpected_characters(e)
        except UnexpectedToken as e:
            handle_unexpected_token(e)

    if DEBUG:
        print(tree.pretty())

    parse_tree(tree)
