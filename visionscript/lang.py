import warnings

warnings.filterwarnings("ignore")

import copy
import io
import logging
import math
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

from visionscript.grammar import grammar
from visionscript.usage import (USAGE, language_grammar_reference,
                   lowercase_language_grammar_reference)

spell = SpellChecker()


def init_state():
    return {
        "last": None,
        "last_function_type": None,
        "last_function_args": None,
        "last_loaded_image": None,
        "image_stack": [],
        "detections_stack": [],
        "history": [],
        # "current_active_model": None,
        "functions": {},
        "output": None,
    }




def get_function_calls():
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
    "variable": lambda x: None,
    "comment": lambda x: None,
    "expr": lambda x: None,
    "show": lambda x: show(x),
    "exit": lambda x: exit(0),
    "help": lambda x: print(language_grammar_reference[x]),
    "train": lambda x: train(x),
    "compare": lambda x: show(x),
    "read": lambda x: read(x),
    "label": lambda x: label(x),
    "list": lambda x: None,
    "get": lambda x: get_func(x),
    "use": lambda x: set_state("current_active_model", x),
    "caption": lambda x: caption(x),
    "contains": lambda x: contains(x),
    "import": lambda x: import_(x)
    }
    return function_calls
# if None, the logic is handled in the main parser

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

def literal_eval(string):
    return string[1:-1] if string.startswith('"') and string.endswith('"') else string

def set_state(key, value):
    global state
    state[key] = value

def make(args):
    global state
    function_name = args[0]

    function_args = args[1:]
    function_calls = get_function_calls()

    function_calls[function_name] = lambda x: None

    state["functions"][function_name] = function_args

def load(filename):
    global state
    import requests
    import validators

    if validators.url(filename):
        response = requests.get(filename)
        file_extension = mimetypes.guess_extension(response.headers["content-type"])

        # if not image, error
        if file_extension not in (".png", ".jpg", ".jpeg"):
            print(f"File {filename} does not represent a png, jpg, or jpeg image.")
            return None

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

    return np.array(Image.open(filename).convert("RGB"))[:, :, ::-1]

def size(_):
    global state
    return state["last_loaded_image"].size

def import_(args):
    # execute code from a file
    # this will update state for the entire script

    file_name = "".join([letter for letter in args if letter.isalpha() or letter == "-" or letter.isdigit()])

    with open(file_name + ".vic", "r") as f:
        code = f.read()
    
    tree = parser.parse(code.strip() + "\n")

    parse_tree(tree)

def cutout(_):
    global state
    x1, y1, x2, y2 = state["last"].xyxy[0]
    image = state["last_loaded_image"]
    cropped_image = image.crop((x1, y1, x2, y2))
    state["image_stack"].append(cropped_image)
    state["last_loaded_image"] = cropped_image

def save(filename):
    global state
    state["last_loaded_image"].save(filename)

def count(args):
    global state
    if len(args) == 0:
        return len(state["last"].xyxy)
    else:
        return len([item for item in state["last"].class_id if item == args[0]])

def detect(classes):
    global state
    logging.disable(logging.CRITICAL)

    from ultralytics import YOLO

    # model name should be only letters and - and numbers

    model_name = state.get("current_active_model", "yolov8n")

    model_name = "".join(
        [
            letter
            for letter in model_name
            if letter.isalpha() or letter == "-" or letter.isdigit()
        ]
    )

    if state.get("model") and state["current_active_model"] == "yolo":
        model = model
    else:
        model = YOLO(model_name + ".pt")

    inference_results = model(state["last_loaded_image"])[0]

    logging.disable(logging.NOTSET)

    # Inference
    results = sv.Detections.from_yolov8(inference_results)

    inference_classes = inference_results.names

    if len(classes) == 0:
        classes = inference_classes

    classes = [key for key, item in inference_classes.items() if item in classes]

    results = results[np.isin(results.class_id, classes)]

    state["detections_stack"].append(results)

    return results

def get_aliased_functions():
    aliased_functions = {
    "isita": "classify",
    "find": "detect",
    "describe": "caption",
    }
    return aliased_functions

def map_alias_to_underlying_function(alias):
    print(alias)
    aliased_functions = get_aliased_functions()
    return aliased_functions.get(alias, alias)

def classify(labels):
    global state
    image = state["last"]

    print(image)

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

    image = (
        preprocess(Image.open(state["last_loaded_image_name"])).unsqueeze(0).to(device)
    )
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # get idx of the most likely label class
        label_idx = probs.argmax()

        label_name = labels[label_idx]

    state["output"] = label_name

    return label_name

def segment(text_prompt):
    global state
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
    global state
    detections = state["last"]

    xyxy = detections.xyxy

    counter = 0

    for i in range(len(xyxy)):
        x1_, y1_, x2_, y2_ = xyxy[i]

        if x1_ >= x1 and y1_ >= y1 and x2_ <= x2 and y2_ <= y2:
            counter += 1

    return counter

def read(_):
    global state
    if state.get("last_function_type", None) in ("detect", "segment"):
        last_args = state["last_function_args"]
        statement = "".join(
            [
                f"{last_args[0]} {state['last'].confidence[i]:.2f} {state['last'].xyxy[i]}\n"
                for i in range(len(state["last"].xyxy))
            ]
        )

        return statement

    return state["last"]

def say(_):
    global state
    if state.get("last_function_type", None) in ("detect", "segment"):
        last_args = state["last_function_args"]
        statement = "".join(
            [
                f"{last_args} {state['last'].confidence[i]:.2f} {state['last'].xyxy[i]}\n"
                for i in range(len(state["last"].xyxy))
            ]
        )
    else:
        statement = state["last"]

    if statement:
        print(statement.strip())

    state["output"] = statement.strip()

def replace(filename):
    global state
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

def label(args):
    global state
    folder = args[0]
    model = args[1]
    items = args[2]

    if "Detect" in state["history"] or state["current_active_model"] == "groundedsam":
        from autodistill.detection import CaptionOntology
        from autodistill_grounded_sam import GroundedSAM

        mapped_items = {item: item for item in items}

        base_model = GroundedSAM(CaptionOntology(mapped_items))
    else:
        print("Please specify a model with which to label images.")
        return

    base_model.label(folder)

def caption(_):
    global state
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    inputs = processor(state["last_loaded_image"], return_tensors="pt")

    out = model.generate(**inputs)

    state["last"] = processor.decode(out[0], skip_special_tokens=True)

    return processor.decode(out[0], skip_special_tokens=True)

def train(args):
    global state
    folder = args[0]
    model = args[1]
    # if Detect or Classify run, train
    if "Detect" in state["history"] or model == "yolov8":
        if "autodistill_yolov8" not in sys.modules:
            from autodistill_yolov8 import YOLOv8

        base_model = YOLOv8("yolov8n.pt")

        model = base_model.train(os.path.join(folder, "data.yaml"), epochs=10)

    elif "Classify" in state["history"] or model == "vit":
        if "autodistill_vit" not in sys.modules:
            import autodistill_vit as ViT

        base_model = ViT("ViT-B/32")

        model = base_model.train(folder, "ViT-B/32")
    else:
        print("No training needed.")
        return

    state["model"] = model

def show(_):
    global state
    # get most recent Detect or Segment
    most_recent_detect_or_segment = None

    for i in range(len(state["history"]) - 1, -1, -1):
        if state["history"][i] in ("detect", "segment"):
            most_recent_detect_or_segment = state["history"][i]
            break

    if most_recent_detect_or_segment == "detect":
        annotator = sv.BoxAnnotator()
    elif most_recent_detect_or_segment == "segment":
        annotator = sv.BoxAnnotator()
    else:
        annotator = None

    if state.get("last_loaded_image_name") is None or not os.path.exists(
        state["last_loaded_image_name"]
    ):
        print("Image does not exist.")
        return

    if state.get("history", [])[-1] == "compare":
        images = []

        grid_size = math.gcd(len(state["image_stack"]), len(state["image_stack"]))

        for image, detections in zip(state["image_stack"], state["detections_stack"]):
            if annotator and detections:
                image = annotator.annotate(np.array(image), detections)
            else:
                image = np.array(image)

            images.append(image)

        sv.plot_images_grid(
            images=np.array(images), grid_size=(grid_size, grid_size), size=(12, 12)
        )

        return

    if annotator:
        image = annotator.annotate(
            cv2.imread(state["last_loaded_image_name"]), state["detections_stack"][-1]
        )
    else:
        image = cv2.imread(state["last_loaded_image_name"])

    if state.get("notebook"):
        buffer = io.BytesIO()
        import base64

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # show image
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        fig.savefig(buffer, format="png")
        buffer.seek(0)

        image = Image.open(buffer)

        state["output"] = {"image": base64.b64encode(buffer.getvalue()).decode("utf-8")}

    sv.plot_image(image, (8, 8))

def get_func(x):
    global state
    state["last"] = state["last"][x]

def contains(statement):
    global state
    if isinstance(state["last"], str):
        return statement in state["last"]
    else:
        return False

def parse_tree(tree, state):
    global DEBUG
    # print(tree)
    if not hasattr(tree, "children"):
        if hasattr(tree, "value") and tree.value.isdigit():
            return int(tree.value)
        elif isinstance(tree, str):
            return literal_eval(tree)

    for node in tree.children:
        # print('e')
        print(node)
        if DEBUG:
            print(node.pretty())

        # ignore EOLs, etc.
        # if node is a list, parse it
        # print all tree attrs
        if node == "True":
            return True
        elif node == "False":
            return False
        elif node is True or node is False:
            return node
        elif hasattr(node, "data") and node.data == "list":
            node = node
        elif (hasattr(node, "type") and node.type == "INT") or isinstance(node, int):
            return int(node.value)
        elif not hasattr(node, "children") or len(node.children) == 0:
            node = node
        elif state.get("ctx") and (state["ctx"].get("in") or state["ctx"].get("if")):
            node = node
        # if string
        elif len(node.children) == 1 and hasattr(node.children[0], "value"):
            return node.children[0].value
        else:
            node = node.children[0]

        if not hasattr(node, "data"):
            continue

        token = node.data
        
        aliased_functions = get_aliased_functions()

        if token.value in aliased_functions:
            token.value = map_alias_to_underlying_function(token.value)

        if token == "comment":
            continue

        if token == "expr":
            parse_tree(node)
            continue

        if token.type == "BOOL":
            return node.children[0].value == "True"

        if token.type == "EQUALITY":
            return parse_tree(node.children[0]) == parse_tree(node.children[1])

        if token == "list":
            results = []

            for item in node.children:
                results.append(parse_tree(item))

            return results

        if token == "var":
            state[node.children[0].children[0].value] = parse_tree(node.children[1])
            state["last"] = state[node.children[0].children[0].value]
            continue

        if token.value == "if":
            # copy state
            last_state_before_if = copy.deepcopy(state)["last"]

            state["ctx"] = {
                "if": True,
            }

            statement = node.children[0]

            statement = parse_tree(statement)

            if statement is None:
                continue

            if statement is False:
                return

            state["last"] = last_state_before_if

        if token.value == "make":
            make(node.children)
            return

        if token.value == None:
            continue

        if token.value == "run":
            function_name = node.children[0].value

            print(f"Running {function_name}...")

            if function_name not in state["functions"]:
                print(f"Function {function_name} does not exist.")
                exit(1)

            function_args = state["functions"][function_name]

            for item in function_args:
                parse_tree(item)

            continue
        function_calls = get_function_calls()

        func = function_calls[token.value]

        if token.value == "get":
            continue

        state["history"].append(token.value)

        if token.value == "say":
            value = state["last"]
            func(value)
            continue
        elif token.value == "contains":
            return func(literal_eval(node.children[0]))
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
                    elif item.type == "INT":
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
        elif all([hasattr(item, "value") for item in node.children]):
            value = [item.value for item in node.children]
        else:
            value = [parse_tree(item) for item in node.children]

        result = func(value)

        if result is not None:
            state["last"] = result

        state["last_function_type"] = token.value
        state["last_function_args"] = [value]

        if token.value == "load":
            state["image_stack"].append(result)
            state["last_loaded_image"] = result

def activate_console(parser):
    print("Welcome to VisionScript!")
    print("Type 'Exit[]' to exit.")
    print("Read the docs at https://visionscript.org/docs")
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


def main() -> None:
    global state, DEBUG

    state = init_state()
    
    parser = Lark(grammar)
    
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--validate", action="store_true", dest="validate", default=False)
    opt_parser.add_option("--ref", action="store_true", dest="ref", default=False)
    opt_parser.add_option("--debug", action="store_true", dest="debug", default=False)
    opt_parser.add_option("--file", action="store", dest="file", default=None)
    opt_parser.add_option("--repl", action="store_true", dest="repl", default=False)

    options, args = opt_parser.parse_args()
        
    if options.validate:
        print("Script is a valid VisionScript program.")
        exit(0)
    
    if options.ref:
        print(USAGE.strip())
    # exit(0)

    if options.debug:
        DEBUG = True
    else:
        DEBUG = False

    if options.file is not None:
        with open(options.file, "r") as f:
            code = f.read() + "\n"
        
        tree = parser.parse(code.lstrip())
        parse_tree(tree, state=state)
    
    
    if options.repl:
        activate_console(parser)
    


if __name__ == "__main__":
    state = init_state()
    main()
    # try:
    #     tree = parser.parse(code.lstrip())
    # except UnexpectedCharacters as e:
    #     handle_unexpected_characters(e)
    # except UnexpectedToken as e:
    #     handle_unexpected_token(e)

    # if DEBUG:
    #     print(tree.pretty())
    #     exit()

    # try:
    #     parse_tree(tree)
    # except KeyboardInterrupt:
    #     print("Exiting VisionScript.")
