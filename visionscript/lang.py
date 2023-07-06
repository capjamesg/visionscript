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
import click
import tempfile

import cv2
import numpy as np
import supervision as sv
from visionscript.grammar import grammar
from lark import Lark, UnexpectedCharacters, UnexpectedToken
from PIL import Image
from spellchecker import SpellChecker
from visionscript.usage import (USAGE, language_grammar_reference,
                   lowercase_language_grammar_reference)

spell = SpellChecker()

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


def literal_eval(string):
    return string[1:-1] if string.startswith('"') and string.endswith('"') else string

def _get_colour_name(rgb_triplet):
    import webcolors

    min_colours = {}
    for key, name in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = key

    return min_colours[min(min_colours.keys())]

    
aliased_functions = {
    "isita": "classify",
    "find": "detect",
    "describe": "caption",
    "getcolors": "getcolours",
}

def map_alias_to_underlying_function(alias):
    return aliased_functions.get(alias, alias)


class VisionScript:
    def __init__(self, notebook=False):
        self.state = {
            "functions": {},
            "last_loaded_image": None,
            "last_loaded_image_name": None,
            "last": None,
            "last_function_type": None,
            "last_function_args": None,
            "image_stack": [],
            "detections_stack": [],
            "history": [],
            #"current_active_model": None,
            "output": None,
        }

        self.function_calls = {
            "load": lambda x: self.load(x),
            "save": lambda x: self.save(x),
            "classify": lambda x: self.classify(x),
            "size": lambda x: self.size(x),
            "say": lambda x: self.say(x),
            "detect": lambda x: self.detect(x),
            "segment": lambda x: self.segment(x),
            "cutout": lambda x: self.cutout(x),
            "count": lambda x: self.count(x),
            "countinregion": lambda x: self.countInRegion(*x),
            "replace": lambda x: self.replace(x),
            "in": lambda x: None,
            "if": lambda x: None,
            "var": lambda x: None,
            "variable": lambda x: None,
            "comment": lambda x: None,
            "expr": lambda x: None,
            "show": lambda x: self.show(x),
            "exit": lambda x: exit(0),
            "help": lambda x: print(language_grammar_reference[x]),
            "train": lambda x: self.train(x),
            "compare": lambda x: self.show(x),
            "read": lambda x: self.read(x),
            "label": lambda x: self.label(x),
            "list": lambda x: None,
            "get": lambda x: self.get_func(x),
            "use": lambda x: self.set_state("current_active_model", x),
            "caption": lambda x: self.caption(x),
            "contains": lambda x: self.contains(x),
            "import": lambda x: self.import_(x),
            "rotate": lambda x: self.rotate(x),
            "getcolours": lambda x: self.getcolours(x),
            "get_text": lambda x: self.get_text(x),
            "greyscale": lambda x: self.greyscale(x),
        }

    def set_state(self, key, value):
        self.state[key] = value


    def make(self, args):
        function_name = args[0]

        function_args = args[1:]

        self.function_calls[function_name] = lambda x: None

        self.state["functions"][function_name] = function_args


    def load(self, filename):
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

        if self.state.get("ctx") and self.state["ctx"].get("in"):
            filename = self.state["ctx"]["active_file"]

        self.state["last_loaded_image_name"] = filename

        return np.array(Image.open(filename).convert("RGB"))[:, :, ::-1]


    def size(self, _):
        return self.state["last_loaded_image"].size


    def import_(self, args):
        # execute code from a file
        # this will update self.state for the entire script

        file_name = "".join(
            [
                letter
                for letter in args
                if letter.isalpha() or letter == "-" or letter.isdigit()
            ]
        )

        with open(file_name + ".vic", "r") as f:
            code = f.read()

        tree = parser.parse(code.strip() + "\n")

        self.parse_tree(tree)


    def cutout(self, _):
        x1, y1, x2, y2 = self.state["last"].xyxy[0]
        image = self.state["last_loaded_image"]
        cropped_image = image.crop((x1, y1, x2, y2))
        self.state["image_stack"].append(cropped_image)
        self.state["last_loaded_image"] = cropped_image


    def select(self, args):
        # if detect, select from detections
        if self.state.get("last_function_type", None) in ("detect", "segment"):
            detections = self.state["last"]

            if len(args) == 0:
                self.state["last"] = detections
            else:
                self.state["last"] = detections[args[0]]


    def paste(self, args):
        x, y = args
        self.state["last_loaded_image"].paste(self.state["image_stack"][-1], (x, y))


    def resize(self, args):
        width, height = args
        image = self.state["last_loaded_image"]
        image = image.resize((width, height))
        self.state["last_loaded_image"] = image


    def pasterandom(self, _):
        x, y = []

        while True:
            x, y = random.randint(0, self.state["last_loaded_image"].size[0]), random.randint(
                0, self.state["last_loaded_image"].size[1]
            )

            if len(self.state["last"].xyxy) == 0:
                break

            for bbox in self.state["last"].xyxy:
                x1, y1, x2, y2 = bbox

                if x1 <= x <= x2 and y1 <= y <= y2:
                    continue

            break

        self.state["last_loaded_image"].paste(self.state["image_stack"][-1], (x, y))


    def save(self, filename):
        self.state["last_loaded_image"].save(filename)


    def count(self, args):
        if len(args) == 0:
            return len(self.state["last"].xyxy)
        else:
            return len([item for item in self.state["last"].class_id if item == args[0]])


    def greyscale(self, _):
        image = self.state["last_loaded_image"]
        image = image.convert("LA")
        self.state["last_loaded_image"] = image


    def get_text(self, _):
        import easyocr

        reader = easyocr.Reader()
        result = reader.readtext(self.state["last_loaded_image_name"])

        return result


    def rotate(self, args):
        image = self.state["last_loaded_image"]
        image = image.rotate(args[0])

        self.state["last_loaded_image"] = image


    def rotate(self, args):
        image = self.state["last_loaded_image"]
        image = image.rotate(args[0])

        self.state["last_loaded_image"] = image


    def getcolours(self, k):
        if not k:
            k = 1

        from sklearn.cluster import KMeans

        image = self.state["last_loaded_image"]

        image = np.array(image)

        image = image.reshape((image.shape[0] * image.shape[1], 3))

        clt = KMeans(n_clusters=k)

        clt.fit(image)

        # map to human readable colour
        centers = clt.cluster_centers_

        human_readable_colours = []

        for center in centers:
            try:
                human_readable_colours.append(
                    _get_colour_name((int(center[0]), int(center[1]), int(center[2])))
                )
            except ValueError as e:
                print(e)
                continue

        self.state["last"] = human_readable_colours[:k]

        return human_readable_colours[:k]


    def detect(self, classes):
        logging.disable(logging.CRITICAL)

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

        if self.state.get("model") and self.state["current_active_model"] == "yolo":
            model = model
        else:
            model = YOLO(model_name + ".pt")

        inference_results = model(self.state["last_loaded_image"])[0]

        logging.disable(logging.NOTSET)

        # Inference
        results = sv.Detections.from_yolov8(inference_results)

        inference_classes = inference_results.names

        if len(classes) == 0:
            classes = inference_classes

        classes = [key for key, item in inference_classes.items() if item in classes]

        results = results[np.isin(results.class_id, classes)]

        self.state["detections_stack"].append(results)

        return results


    def classify(self, labels):
        image = self.state["last"]

        if self.state.get("model") and self.state["model"].__class__.__name__ == "ViT":
            model = self.state["model"]

            results = model.predict(image).get_top_k(1)

            if len(results.class_id) == 0:
                return sv.Classifications.empty()

            return results.class_id[0]
        elif self.state.get("model") and self.state["model"].__class__.__name__ == "YOLOv8":
            model = self.state["model"]

            results = model.predict(image)

            return results

        if "clip" not in sys.modules:
            import clip
            import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        image = (
            preprocess(Image.open(self.state["last_loaded_image_name"])).unsqueeze(0).to(device)
        )
        text = clip.tokenize(labels).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # get idx of the most likely label class
            label_idx = probs.argmax()

            label_name = labels[label_idx]

        self.state["output"] = label_name

        return label_name


    def segment(self, text_prompt):
        if "FastSAM" not in sys.modules:
            from fastsam import FastSAM, FastSAMPrompt

        logging.disable(logging.CRITICAL)
        model = FastSAM("./weights/FastSAM.pt")

        DEVICE = "cpu"
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

        return sv.Detections(
            mask=np.array([item.mask[0] for item in results]),
            xyxy=np.array([item.xyxy[0] for item in results]),
            class_id=np.array(class_ids),
            confidence=np.array([1]),
        )


    def countInRegion(self, x1, y1, x2, y2):
        detections = self.state["last"]

        xyxy = detections.xyxy

        counter = 0

        for i in range(len(xyxy)):
            x1_, y1_, x2_, y2_ = xyxy[i]

            if x1_ >= x1 and y1_ >= y1 and x2_ <= x2 and y2_ <= y2:
                counter += 1

        return counter


    def read(self, _):
        if self.state.get("last_function_type", None) in ("detect", "segment"):
            last_args = self.state["last_function_args"]
            statement = "".join(
                [
                    f"{last_args[0]} {state['last'].confidence[i]:.2f} {state['last'].xyxy[i]}\n"
                    for i in range(len(self.state["last"].xyxy))
                ]
            )

            return statement

        return self.state["last"]


    def say(self, _):
        if self.state.get("last_function_type", None) in ("detect", "segment"):
            last_args = self.state["last_function_args"]
            statement = "".join(
                [
                    f"{last_args} {self.state['last'].confidence[i]:.2f} {state['last'].xyxy[i]}\n"
                    for i in range(len(self.state["last"].xyxy))
                ]
            )
        elif isinstance(self.state["last"], list):
            statement = ", ".join([str(item) for item in self.state["last"]])
        else:
            statement = self.state["last"]

        if statement:
            print(statement.strip())

        self.state["output"] = statement.strip()


    def replace(self, filename):
        detections = self.state["last"]

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
        self.state["last_loaded_image"].paste(random_img, (int(xyxy[0][0]), int(xyxy[0][1])))


    def label(self, args):
        folder = args[0]
        model = args[1]
        items = args[2]

        if "Detect" in self.state["history"] or self.state["current_active_model"] == "groundedsam":
            from autodistill.detection import CaptionOntology
            from autodistill_grounded_sam import GroundedSAM

            mapped_items = {item: item for item in items}

            base_model = GroundedSAM(CaptionOntology(mapped_items))
        else:
            print("Please specify a model with which to label images.")
            return

        base_model.label(folder)


    def caption(self, _):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        inputs = processor(self.state["last_loaded_image"], return_tensors="pt")

        out = model.generate(**inputs)

        self.state["last"] = processor.decode(out[0], skip_special_tokens=True)

        return processor.decode(out[0], skip_special_tokens=True)


    def train(self, args):
        folder = args[0]
        model = args[1]
        # if Detect or Classify run, train
        if "Detect" in self.state["history"] or model == "yolov8":
            if "autodistill_yolov8" not in sys.modules:
                from autodistill_yolov8 import YOLOv8

            base_model = YOLOv8("yolov8n.pt")

            model = base_model.train(os.path.join(folder, "data.yaml"), epochs=10)

        elif "Classify" in self.state["history"] or model == "vit":
            if "autodistill_vit" not in sys.modules:
                import autodistill_vit as ViT

            base_model = ViT("ViT-B/32")

            model = base_model.train(folder, "ViT-B/32")
        else:
            print("No training needed.")
            return

        self.state["model"] = model


    def show(self, _):
        # get most recent Detect or Segment
        most_recent_detect_or_segment = None

        for i in range(len(self.state["history"]) - 1, -1, -1):
            if self.state["history"][i] in ("detect", "segment"):
                most_recent_detect_or_segment = self.state["history"][i]
                break

        if most_recent_detect_or_segment == "detect":
            annotator = sv.BoxAnnotator()
        elif most_recent_detect_or_segment == "segment":
            annotator = sv.BoxAnnotator()
        else:
            annotator = None

        if self.state.get("last_loaded_image_name") is None or not os.path.exists(
            self.state["last_loaded_image_name"]
        ):
            print("Image does not exist.")
            return

        if self.state.get("history", [])[-1] == "compare":
            images = []

            grid_size = math.gcd(len(self.state["image_stack"]), len(self.state["image_stack"]))

            for image, detections in zip(self.state["image_stack"], self.state["detections_stack"]):
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
                cv2.imread(self.state["last_loaded_image_name"]), self.state["detections_stack"][-1]
            )
        else:
            image = cv2.imread(self.state["last_loaded_image_name"])

        if self.state.get("notebook"):
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

            self.state["output"] = {"image": base64.b64encode(buffer.getvalue()).decode("utf-8")}

        sv.plot_image(image, (8, 8))


    def get_func(self, x):
        self.state["last"] = self.state["last"][x]


    def contains(self, statement):
        if isinstance(self.state["last"], str):
            return statement in self.state["last"]
        else:
            return False


    def parse_tree(self, tree):
        # print(tree)
        if not hasattr(tree, "children"):
            if hasattr(tree, "value") and tree.value.isdigit():
                return int(tree.value)
            elif isinstance(tree, str):
                return literal_eval(tree)

        for node in tree.children:
            print(node)
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
            elif self.state.get("ctx") and (self.state["ctx"].get("in") or self.state["ctx"].get("if")):
                node = node
            # if string
            elif len(node.children) == 1 and hasattr(node.children[0], "value"):
                return node.children[0].value
            else:
                node = node.children[0]

            if not hasattr(node, "data"):
                continue

            token = node.data

            if token.value in aliased_functions:
                token.value = map_alias_to_underlying_function(token.value)

            if token == "comment":
                continue

            if token == "expr":
                self.parse_tree(node)
                continue

            if token.type == "BOOL":
                return node.children[0].value == "True"

            if token.type == "EQUALITY":
                return self.parse_tree(node.children[0]) == parse_tree(node.children[1])

            if token == "list":
                results = []

                for item in node.children:
                    results.append(self.parse_tree(item))

                return results

            if token == "var":
                self.state[node.children[0].children[0].value] = parse_tree(node.children[1])
                self.state["last"] = self.state[node.children[0].children[0].value]
                continue

            if token.value == "if":
                # copy self.state
                last_state_before_if = copy.deepcopy(state)["last"]

                self.state["ctx"] = {
                    "if": True,
                }

                statement = node.children[0]

                statement = self.parse_tree(statement)

                if statement is None:
                    continue

                if statement is False:
                    return

                self.state["last"] = last_state_before_if

            if token.value == "make":
                self.make(node.children)
                return

            if token.value == None:
                continue

            if token.value == "run":
                function_name = node.children[0].value

                print(f"Running {function_name}...")

                if function_name not in self.state["functions"]:
                    print(f"Function {function_name} does not exist.")
                    exit(1)

                function_args = self.state["functions"][function_name]

                for item in function_args:
                    self.parse_tree(item)

                continue

            func = self.function_calls[token.value]

            if token.value == "get":
                continue

            self.state["history"].append(token.value)

            if token.value == "say":
                value = self.state["last"]
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
                self.state["ctx"] = {
                    "in": os.listdir(node.children[0].value),
                }

                for file_name in self.state["ctx"]["in"]:
                    self.state["ctx"]["active_file"] = os.path.join(
                        literal_eval(node.children[0]), file_name
                    )
                    # ignore first 2, then do rest
                    context = node.children[3:]

                    for item in context:
                        self.parse_tree(item)

                del self.state["ctx"]

                continue

            if len(node.children) == 1:
                value = node.children[0].value
            elif all([hasattr(item, "value") for item in node.children]):
                value = [item.value for item in node.children]
            else:
                value = [self.parse_tree(item) for item in node.children]

            result = func(value)

            if result is not None:
                self.state["last"] = result

            self.state["last_function_type"] = token.value
            self.state["last_function_args"] = [value]

            if token.value == "load":
                self.state["image_stack"].append(result)
                self.state["last_loaded_image"] = result


def activate_console(parser):
    print("Welcome to VisionScript!")
    print("Type 'Exit[]' to exit.")
    print("Read the docs at https://visionscript.org/docs")
    print("For help, type 'Help[FunctionName]'.")
    print("-" * 20)
    session = VisionScript()

    while True:
        code = input(">>> ")

        try:
            tree = parser.parse(code.lstrip())
        except UnexpectedCharacters as e:
            handle_unexpected_characters(e)
        except UnexpectedToken as e:
            handle_unexpected_token(e)

        session.parse_tree(tree)


@click.command()
@click.option("--validate", default=False, help="")
@click.option("--ref", default=False, help="Name of the file")
@click.option("--debug", default=False, help="To debug")
@click.option("--file", default=None, help="Name of the file")
@click.option("--repl", default=None, help="To enter to vscript console")
def main(validate, ref, debug, file, repl) -> None:
    parser = Lark(grammar)
    
    if validate:
        print("Script is a valid VisionScript program.")
        exit(0)
    
    if ref:
        print(USAGE.strip())

    if debug:
        DEBUG = True
    else:
        DEBUG = False

    if file is not None:
        with open(file, "r") as f:
            code = f.read() + "\n"
        
        tree = parser.parse(code.lstrip())

        session = VisionScript()

        session.parse_tree(tree)
    
    
    if repl == 'console':
        activate_console(parser)
    
if __name__ == "__main__":
    main()
