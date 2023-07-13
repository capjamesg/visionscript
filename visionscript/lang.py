import warnings

warnings.filterwarnings("ignore")

import copy
import io
import logging
import math
import mimetypes
import os
import random
import string
import sys
import tempfile
import time

import click
import cv2
import lark
import numpy as np
import supervision as sv
import torch
from lark import Lark, UnexpectedCharacters, UnexpectedToken
from PIL import Image
from spellchecker import SpellChecker

from visionscript.grammar import grammar
from visionscript.usage import (USAGE, language_grammar_reference,
                                lowercase_language_grammar_reference)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FILE_SIZE = 10000000  # 10MB

spell = SpellChecker()

parser = Lark(grammar)

SUPPORTED_INFERENCE_MODELS = {
    "classify": {
        "clip": "clip",
    },
    "detect": {"yolov8": "autodistill_yolov8"},
    "segment": {"fastsam": "fastsam", "groundedsam": "autodistill_groundedsam"},
}

SUPPORTED_TRAIN_MODELS = {
    "classify": {"vit": "autodistill_vit"},
    "detect": {"yolov8": "autodistill_yolov8"},
}


class InputNotProvided:
    pass


def handle_unexpected_characters(e, code):
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


def init_state():
    return {
        "functions": {},
        "last_loaded_image": None,
        "last_loaded_image_name": None,
        "last": None,
        "last_function_type": None,
        "last_function_args": None,
        "image_stack": [],
        "detections_stack": [],
        "history": [],
        "search_index_stack": [],
        # "current_active_model": None,
        "output": None,
        "input_variables": {},
        "last_classes": [],
        "confidence": 50,
        "active_region": None,
        "active_filters": {"class": None, "region": None},
    }


class VisionScript:
    """
    A VisionScript program.
    """

    def __init__(self, notebook=False):
        self.state = init_state()
        self.notebook = notebook
        self.code = ""

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
            "select": lambda x: self.select(x),
            "paste": lambda x: self.paste(x),
            "pasterandom": lambda x: self.pasterandom(x),
            "resize": lambda x: self.resize(x),
            "blur": lambda x: self.blur(x),
            "make": lambda x: self.make(x),
            "args": lambda x: None,
            "setbrightness": lambda x: self.set_brightness(x),
            "search": lambda x: self.search(x),
            "similarity": lambda x: self.similarity(x),
            "readqr": lambda x: self.read_qr(x),
            "reset": lambda x: self.reset(x),
            "negate": lambda x: self.negate(x),
            "equality": lambda x: self.equality(x),
            "not_equality": lambda x: not self.equality(x),
            "input": lambda x: self.input_(x),
            "deploy": lambda x: self.deploy(x),
            "getedges": lambda x: self.get_edges(x),
            "setregion": lambda x: self.set_region(x),
            "setconfidence": lambda x: self.set_confidence(x),
            "filterbyclass": lambda x: self.filter_by_class(x),
        }

    def filter_by_class(self, args):
        """
        Filter detections by class.
        """
        if len(args) == 0:
            self.state["active_filters"] = None
        elif self.state["active_filters"].get("class") is None:
            self.state["active_filters"]["class"] = args

    def set_region(self, args):
        if len(args) == 0:
            self.state["active_filters"]["region"] = None
            return

        x0, y0, x1, y1 = args

        self.state["active_filters"]["region"] = (x0, y0, x1, y1)

    def input_(self, key):
        if self.state["input_variables"].get(literal_eval(key)) is not None:
            return self.state["input_variables"][literal_eval(key)]
        else:
            return InputNotProvided()
            # print(f"Input {key} does not exist.")
            # exit()

    def equality(self, args):
        return args[0] == args[1]

    def negate(self, expr):
        return not expr

    def reset(self, _):
        self.state = init_state()

    def set_state(self, key, value):
        self.state[key] = value

    def make(self, args):
        """
        Declare a function.
        """
        function_name = args[0].children[0].value.strip()

        function_body = lark.Tree("expr", args[1:])

        self.state["functions"][function_name] = function_body

    def set_confidence(self, confidence):
        """
        Set the confidence level for use in filtering detections.
        """
        if not confidence:
            confidence = 50

        if confidence > 100 or confidence < 0:
            print("Confidence must be between 0 and 100.")
            return

        self.state["confidence"] = confidence

    def load(self, filename):
        """
        Load an image or folder of images into state.
        """
        import requests
        import validators

        if isinstance(filename, np.ndarray):
            self.state["image_stack"].append(filename)
            # save file
            import uuid

            name = str(uuid.uuid4()) + ".png"
            cv2.imwrite(name, filename)

            self.state["last_loaded_image_name"] = name

            return filename

        # if is dir, load all images
        if filename and os.path.isdir(filename):
            image_filenames = [filename + "/" + item for item in os.listdir(filename)]

            for image_filename in image_filenames:
                self.load(image_filename)

            return

        if filename and validators.url(filename):
            try:
                response = requests.get(filename, timeout=5, stream=True)
            except requests.exceptions.ConnectionError:
                self.state["last"] = "Could not connect to URL."
                return
            except requests.exceptions.ReadTimeout:
                self.state["last"] = "Timeout."
                return
            except:
                self.state["last"] = "There was an error loading the image."
                return

            # check length
            if len(response.content) > MAX_FILE_SIZE:
                self.state["last"] = "Image too large."
                return

            size = 0
            start = time.time()

            for chunk in response.iter_content(1024):
                if time.time() - start > 5:
                    raise ValueError("timeout reached")

                size += len(chunk)

                if size > MAX_FILE_SIZE:
                    raise ValueError("response too large")

            file_extension = mimetypes.guess_extension(response.headers["content-type"])

            # if not image, error
            if file_extension not in (".png", ".jpg", ".jpeg"):
                print(f"File {filename} does not represent a png, jpg, or jpeg image.")
                return None

            # 10 random characters
            filename = (
                "".join(
                    random.choice(string.ascii_letters + string.digits)
                    for _ in range(10)
                )
                + file_extension
            )

            with tempfile.NamedTemporaryFile(delete=True) as f:
                f.write(response.content)
                filename = f.name

        if self.state.get("ctx") and self.state["ctx"].get("in"):
            filename = self.state["ctx"]["active_file"]

        # make filename safe
        # from werkzeug.utils import secure_filename

        # filename = filename.split("/")[-1]

        # filename = secure_filename(filename)

        self.state["last_loaded_image_name"] = filename

        self.state["output"] = {"image": Image.open(filename).convert("RGB")}

        return np.array(Image.open(filename).convert("RGB"))  # [:, :, ::-1]

    def size(self, _):
        return self.state["image_stack"][-1].shape[:2]

    def import_(self, args):
        """
        Import a module from another file.
        """
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
        """
        Cut out a detection from an image.
        """
        x1, y1, x2, y2 = self.state["last"].xyxy[0]
        image = self.state["image_stack"][-1]
        cropped_image = image.crop((x1, y1, x2, y2))
        self.state["image_stack"].append(cropped_image)

    def select(self, args):
        """
        Select a detection from a sv.Detections object.
        """
        # if detect, select from detections
        if self.state.get("last_function_type", None) in (
            "detect",
            "segment",
            "classify",
        ):
            detections = self.state["last"]

            detections = self._filter_controller(detections)

            if len(args) == 0:
                self.state["last"] = detections
            else:
                self.state["last"] = detections[args[0]]

    def paste(self, args):
        """
        Paste an image onto another image.
        """
        x, y = args
        self.state["image_stack"].append(
            self.state["image_stack"][-2].paste(self.state["image_stack"][-1], (x, y))
        )

    def resize(self, args):
        """
        Resize an image.
        """
        width, height = args
        image = self.state["image_stack"][-1]
        image = cv2.resize(image, (width, height))
        self.state["image_stack"].append(image)

    def _create_index(self):
        import faiss

        index = faiss.IndexFlatL2(512)

        self.state["search_index_stack"].append(index)

        return index

    def _add_to_index(self, image):
        index = self.state["search_index_stack"][-1]

        index.add(image)

    def search(self, label):
        """
        Search for an image using a text label or image.

        On first run, this will create an index of all images on the loaded image stack.
        """
        import clip

        model, preprocess = clip.load("ViT-B/32", device=DEVICE)

        with torch.no_grad():
            # if label is a filename, load image
            if os.path.exists(label):
                comparator = preprocess(Image.open(label)).unsqueeze(0).to(DEVICE)

                comparator = model.encode_image(comparator)
            else:
                comparator = clip.tokenize([label]).to(DEVICE)

                comparator = model.encode_text(comparator)

            if len(self.state["search_index_stack"]) == 0:
                self._create_index()

                for image in self.state["image_stack"]:
                    # turn cv2 image into PIL image
                    image = Image.fromarray(image)

                    processed_image = preprocess(image).unsqueeze(0).to(DEVICE)
                    embedded_image = model.encode_image(processed_image)

                    self._add_to_index(embedded_image)

        index = self.state["search_index_stack"][-1]

        results = index.search(comparator, 5)

        image_names = []

        for result in results[1][0]:
            image_names.append(self.state["image_stack"][result])

        if len(self.state["image_stack"]) < 5:
            image_names = image_names[: len(self.state["image_stack"])]

        return image_names

    def pasterandom(self, _):
        """
        Paste the most recent image in a random position on the previously most recent image.
        """
        x, y = []

        while True:
            x, y = random.randint(0, self.state["image_stack"].size[0]), random.randint(
                0, self.state["image_stack"].size[1]
            )

            if len(self.state["last"].xyxy) == 0:
                break

            for bbox in self.state["last"].xyxy:
                x1, y1, x2, y2 = bbox

                if x1 <= x <= x2 and y1 <= y <= y2:
                    continue

            break

        self.state["image_stack"][-1] = self.state["image_stack"][-2].paste(
            self.state["image_stack"][-1], (x, y)
        )

    def save(self, filename):
        """
        Save an image to a file.
        """
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

        if not os.path.exists("tmp/output"):
            os.makedirs("tmp/output")

        if not filename:
            filename = os.path.join(
                "tmp/output/",
                "".join(random.choice(string.ascii_letters) for _ in range(10)),
            )

        self.state["image_stack"].save(filename)

        self.state["output"] = {"text": "Saved to " + filename}

    def count(self, args):
        """
        Count the number of detections in a sv.Detections object.
        """
        if len(args) == 0:
            return len(self.state["last"].xyxy)
        else:
            return len(
                [item for item in self.state["last"].class_id if item == args[0]]
            )

    def greyscale(self, _):
        """
        Turn an image to greyscale.
        """
        image = self.state["image_stack"][-1]
        # turn to bgr
        image = image[:, :, ::-1].copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.state["image_stack"].append(image)
        # save to test.png

        self.state["image_stack"].append(image)
        self.state["output"] = {"image": image}

    def deploy(self, app_name):
        """
        Deploy a script to a VisionScript Cloud web app.
        """
        # make POST to http://localhost:6999/create
        import string

        import requests

        app_slug = app_name.translate(
            str.maketrans("", "", string.punctuation.replace("-", ""))
        ).replace(" ", "-")

        response = requests.post(
            "http://localhost:6999/create",
            json={
                "api_key": "test",
                "title": app_name,
                "slug": app_slug,
                "script": self.code,
                "variables": self.state["input_variables"],
            },
        )

        if response.ok:
            return "Deployed successfully to http://localhost:6999/app"

        return "Error deploying."

    def get_text(self, _):
        """
        Use OCR to get text from an image.
        """
        import easyocr

        reader = easyocr.Reader(["en"])
        result = reader.readtext(self.state["last_loaded_image_name"], detail=0)

        return result

    def rotate(self, args):
        """
        Rotate an image.
        """
        image = self.state["image_stack"][-1]
        # load into cv2
        args = int(args)
        if args == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif args == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif args == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            image = image

        self.state["output"] = {"image": image}
        self.state["image_stack"].append(image)

    def getcolours(self, k):
        """
        Get the most common colours in an image.
        """
        if not k:
            k = 1

        from sklearn.cluster import KMeans

        image = self.state["image_stack"][-1]

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
                continue

        self.state["last"] = human_readable_colours[:k]

        return human_readable_colours[:k]

    def _filter_controller(self, detections):
        results = detections

        if self.state["active_filters"]["region"]:
            x0, y0, x1, y1 = self.state["active_filters"]["region"]

            zone = sv.PolygonZone(
                np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]),
                frame_resolution_wh=(
                    self.state["last"].shape[1],
                    self.state["last"].shape[0],
                ),
            )

            results_filtered_by_active_region = zone.trigger(detections=results)

            results = results[results_filtered_by_active_region]

        if self.state["active_filters"]["class"]:
            results = results[
                np.isin(results.class_id, self.state["active_filters"]["class"])
            ]

        results = results[results.confidence > self.state["confidence"] / 100]

        # self.state["last"] = results

        return results

    def detect(self, classes):
        """
        Run object detection on an image.
        """
        logging.disable(logging.CRITICAL)

        if (
            self.state.get("current_active_model")
            and self.state["current_active_model"].lower() == "groundingdino"
        ):
            from autodistill.detection import CaptionOntology
            from autodistill_grounding_dino import GroundingDINO

            mapped_items = {item: item for item in classes}

            base_model = GroundingDINO(CaptionOntology(mapped_items))

            inference_results = base_model.predict(self.state["last_loaded_image_name"])
        else:
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

            if (
                self.state.get("model")
                and self.state["current_active_model"].lower() == "yolo"
            ):
                model = model
            else:
                model = YOLO(model_name + ".pt")

            inference_results = model(self.state["image_stack"][-1])[0]

            logging.disable(logging.NOTSET)

            # Inference
            results = sv.Detections.from_yolov8(inference_results)

        inference_classes = inference_results.names

        if len(classes) == 0:
            classes = inference_classes

        classes = [key for key, item in inference_classes.items() if item in classes]

        results = results[np.isin(results.class_id, classes)]

        self.state["detections_stack"].append(results)
        self.state["last_classes"] = [inference_classes[item] for item in classes]

        return results

    def classify(self, labels):
        """
        Classify an image using provided labels.
        """
        image = self.state["last"]

        if self.state.get("model") and self.state["model"].__class__.__name__ == "ViT":
            model = self.state["model"]

            results = model.predict(image).get_top_k(1)

            if len(results.class_id) == 0:
                return sv.Classifications.empty()

            return results.class_id[0]
        elif (
            self.state.get("model")
            and self.state["model"].__class__.__name__ == "YOLOv8"
        ):
            model = self.state["model"]

            results = model.predict(image)

            return results

        import clip

        model, preprocess = clip.load("ViT-B/32", device=DEVICE)

        image = (
            preprocess(Image.open(self.state["last_loaded_image_name"]))
            .unsqueeze(0)
            .to(DEVICE)
        )
        text = clip.tokenize(labels).to(DEVICE)

        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # get idx of the most likely label class
            label_idx = probs.argmax()

            label_name = labels[label_idx]

        self.state["output"] = {"text": label_name}

        return label_name

    def segment(self, text_prompt):
        """
        Apply image segmentation and generate segmentation masks.
        """
        # check for model
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

        self.state["detections_stack"].append(detections)

        return detections

    def read(self, _):
        if self.state.get("last_function_type", None) in ("detect", "segment"):
            last_args = self.state["last_function_args"]
            statement = "".join(
                [
                    f"{last_args[0]} {self.state['last'].confidence[i]:.2f} {self.state['last'].xyxy[i]}\n"
                    for i in range(len(self.state["last"].xyxy))
                ]
            )

            return statement

        return self.state["last"]

    def say(self, statement):
        """
        Print a statement to the console, or create a text representation of a statement for use
        in a notebook.
        """
        if isinstance(self.state["last"], np.ndarray):
            self.show(None)
            return

        if isinstance(self.state["last"], (list, tuple)):
            output = ""

            for item in self.state["last"]:
                # if list or tuple, join
                if isinstance(item, (list, tuple)):
                    item = ", ".join([str(i) for i in item])
                elif isinstance(item, int) or isinstance(item, float):
                    item = str(item)

                output += item + "\n"

            self.state["output"] = {"text": output}

            return

        if isinstance(statement, int) or isinstance(statement, float):
            statement = str(statement)

        if statement and isinstance(statement, str):
            print(statement.strip())
            return

        if self.state.get("last_function_type", None) in ("detect", "segment"):
            last_args = self.state["last_function_args"]
            statement = "".join(
                [
                    f"{last_args} {self.state['last'].confidence[i]:.2f} {self.state['last'].xyxy[i]}\n"
                    for i in range(len(self.state["last"].xyxy))
                ]
            )
        elif isinstance(self.state["last"], list):
            statement = ", ".join([str(item) for item in self.state["last"]])
        else:
            statement = self.state["last"]

        if statement:
            print(statement.strip())

        self.state["output"] = {"text": statement}

    def blur(self, args):
        """
        Blur an image.
        """
        image = self.state["image_stack"][-1]

        image = cv2.blur(image, (args[0], args[0]))

        self.state["image_stack"].append(image)

    def replace(self, args):
        """
        Replace a detection or list of detections with an image or color.
        """
        detections = self.state["last"]

        detections = self._filter_controller(detections)

        xyxy = detections.xyxy

        if os.path.exists(args):
            print("Replacing with image.")
            picture = cv2.imread(args)

            # bgr to rgb
            picture = picture[:, :, ::-1].copy()

            image_at_top_of_stack = self.state["image_stack"][-1].copy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # resize picture to fit
                picture = cv2.resize(picture, (x2 - x1, y2 - y1))

                image_at_top_of_stack[y1:y2, x1:x2] = picture

            self.state["image_stack"].append(image_at_top_of_stack)

            return

        color = args

        if args is not None:
            import webcolors

            try:
                color_to_rgb = webcolors.name_to_rgb(color)
            except ValueError:
                print(f"Color {color} does not exist.")
                return

            color_to_rgb = np.array(color_to_rgb)

            # convert to bgr
            color_to_rgb = color_to_rgb[::-1]

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]

                # cast all to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                self.state["image_stack"][-1][y1:y2, x1:x2] = color_to_rgb

    def label(self, args):
        """
        Label a folder of images using an object detection, classification, or segmentation model.

        Not fully implemented.
        """
        folder = args[0]
        items = args[2]

        if (
            "Detect" in self.state["history"]
            or self.state["current_active_model"] == "groundedsam"
        ):
            from autodistill.detection import CaptionOntology
            from autodistill_grounded_sam import GroundedSAM

            mapped_items = {item: item for item in items}

            base_model = GroundedSAM(CaptionOntology(mapped_items))
        else:
            print("Please specify a model with which to label images.")
            return

        base_model.label(folder)

    def caption(self, _):
        """
        Generate a caption for an image.
        """
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        inputs = processor(self.state["image_stack"][-1], return_tensors="pt")

        out = model.generate(**inputs)

        self.state["last"] = processor.decode(out[0], skip_special_tokens=True)

        return processor.decode(out[0], skip_special_tokens=True)

    def train(self, args):
        """
        Train a model on a dataset.

        Not fully implemented.
        """
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

    def get_edges(self, _):
        """
        Convert image to greyscale then perform Sobel edge detection.
        """
        self.greyscale(_)

        image = self.state["image_stack"][-1]

        sobelxy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)

        self.state["image_stack"].append(sobelxy)
        self.state["output"] = {"image": sobelxy}

    def show(self, _):
        """
        Show the image in the notebook or in a new window.

        If a Detect or Segment function was run previously, this function shows the image with the bounding boxes.
        """
        most_recent_detect_or_segment = None

        if self.state["history"][-2] in ("detect", "segment"):
            most_recent_detect_or_segment = self.state["history"][-2]

        if most_recent_detect_or_segment == "detect":
            annotator = sv.BoxAnnotator()
        elif most_recent_detect_or_segment == "segment":
            annotator = sv.MaskAnnotator()
        else:
            annotator = None

        if self.state.get("last_loaded_image_name") is None or not os.path.exists(
            self.state["last_loaded_image_name"]
        ):
            print("Image does not exist.")
            return

        if self.state.get("history", [])[-2] == "search":
            images = []

            grid_size = math.gcd(len(self.state["last"]), len(self.state["last"]))

            if len(self.state["last"]) == len(self.state["detections_stack"]):
                for image, detections in zip(
                    self.state["last"], self.state["detections_stack"]
                ):
                    detections = self._filter_controller(self, detections)
                    if annotator and detections:
                        image = annotator.annotate(
                            np.array(image),
                            detections,
                            labels=self.state["last_classes"],
                        )
                    else:
                        image = np.array(image)

                    images.append(image)
            else:
                for image in self.state["last"]:
                    images.append(np.array(image))

            if not self.notebook:
                sv.plot_images_grid(
                    images=np.array(images),
                    grid_size=(grid_size, grid_size),
                    size=(12, 12),
                )

                return

            image = images[0]

        elif self.state.get("history", [])[-1] == "compare":
            images = []

            grid_size = math.gcd(
                len(self.state["image_stack"]), len(self.state["image_stack"])
            )

            for image, detections in zip(
                self.state["image_stack"], self.state["detections_stack"]
            ):
                detections = self._filter_controller(self, detections)
                if annotator and detections:
                    image = annotator.annotate(
                        np.array(image), detections, labels=self.state["last_classes"]
                    )
                else:
                    image = np.array(image)

                images.append(image)

            if not self.notebook:
                sv.plot_images_grid(
                    images=np.array(images),
                    grid_size=(grid_size, grid_size),
                    size=(12, 12),
                )

                return

            # image = images[0]

        if annotator:
            image = annotator.annotate(
                np.array(self.state["image_stack"][-1]),
                detections=self._filter_controller(self.state["detections_stack"][-1]),
            )
        elif (
            self.state.get("last_loaded_image") is not None
            and not self.state.get("history", [])[-1] == "compare"
        ):
            image = self.state["image_stack"][-1]
        else:
            image = self.state["image_stack"][-1]
        # elif self.notebook is False:
        #     image = cv2.imread(self.state["last_loaded_image_name"])

        if self.notebook:
            buffer = io.BytesIO()
            import base64

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # show image
            if annotator:
                fig = plt.figure(figsize=(8, 8))
                # if grey, show in grey
                if len(image.shape) == 2:
                    plt.imshow(image, cmap="gray")
                # if bgr, show in rgb
                elif image.shape[2] == 3:
                    plt.imshow(image[:, :, ::-1])

                fig.savefig(buffer, format="png")
                buffer.seek(0)

                image = Image.open(buffer)

                img_str = {"image": base64.b64encode(buffer.getvalue()).decode("utf-8")}

                self.state["output"] = {"image": img_str}

                return
            else:
                # if ndarray, convert to PIL
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                    # do bgr to rgb
                    # image = image.convert("RGB")

                # # convert to rgb if needed
                # if image.mode != "RGB":
                #     image = image.convert("RGB")

                # PIL to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            self.state["output"] = {"image": img_str}

            return

        sv.plot_image(image, (8, 8))

    def get_func(self, x):
        self.state["last"] = self.state["last"][x]

    def similarity(self, n):
        """
        Get similarity of last N images.
        """
        if not n:
            n = 2

        if len(self.state["image_stack"]) < 2 or len(self.state["image_stack"]) < n:
            print("Not enough images to compare.")
            return

        import clip

        model, preprocess = clip.load("ViT-B/32", device=DEVICE)

        images = []

        for image in self.state["image_stack"][-n:]:
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(DEVICE)
            images.append(image)

        embeddings = []

        with torch.no_grad():
            for image in images:
                image = model.encode_image(image)

                embeddings.append(image)

        # get similarity
        similarity = torch.cosine_similarity(embeddings[0], embeddings[1])

        # cast tensor to float
        as_float = similarity.item()

        self.state["last"] = as_float

        return as_float

    def read_qr(self, _):
        """
        Read QR code from last image.
        """
        image = self.state["image_stack"][-1]

        data, _, _ = cv2.QRCodeDetector().detectAndDecode(image)

        return data

    def set_brightness(self, brightness):
        """
        Set brightness of last image.
        """
        # brightness is between -100 and 100
        image = self.state["image_stack"][-1]

        # use cv2
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # add brightness
        lim = 255 - brightness

        v[v > lim] = 255
        v[v <= lim] += brightness

        final_hsv = cv2.merge((h, s, v))

        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        self.state["image_stack"].append(image)
        self.state["output"] = {"image": image}

    def contains(self, statement):
        """
        Check if a statement is contained in the last statement.
        """
        if isinstance(self.state["last"], str):
            return statement in self.state["last"]
        else:
            return False

    def check_inputs(self, tree):
        # get all INPUT tokens and save them to state
        for node in tree.children:
            # if node has children, recurse
            if hasattr(node, "children") and len(node.children) > 0:
                self.check_inputs(node)

            if not hasattr(node, "data"):
                continue

            if node.data == "input":
                self.state["input_variables"][node.children[0].value] = "image"

    def parse_tree(self, tree):
        """
        Abstract Syntax Tree (AST) parser for VisionScript.
        """

        if not hasattr(tree, "children"):
            if hasattr(tree, "value") and tree.value.isdigit():
                return int(tree.value)
            elif isinstance(tree, str):
                return literal_eval(tree)
            elif hasattr(tree, "value") and tree.value.isfloat():
                return float(tree.value)

        if hasattr(tree, "children") and tree.data == "input":
            return self.input_(tree.children[0].value)

        for node in tree.children:
            if node == "True":
                return True
            elif node == "False":
                return False
            # if equality, check if equal
            # if rule is input
            elif hasattr(node, "data") and node.data == "equality":
                return self.parse_tree(node.children[0]) == self.parse_tree(
                    node.children[1]
                )
            elif node is True or node is False:
                return node
            elif hasattr(node, "data") and node.data == "list":
                node = node
            elif (hasattr(node, "type") and node.type == "INT") or isinstance(
                node, int
            ):
                return int(node.value)
            elif (hasattr(node, "type") and node.type == "FLOAT") or isinstance(
                node, float
            ):
                return float(node.value)
            elif not hasattr(node, "children") or len(node.children) == 0:
                node = node
            elif self.state.get("ctx") and (
                self.state["ctx"].get("in") or self.state["ctx"].get("if")
            ):
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

            if token.type == "equality":
                return self.parse_tree(node.children[0]) == self.parse_tree(
                    node.children[1]
                )

            if token == "comment":
                continue

            if token == "expr":
                self.parse_tree(node)
                continue

            if token.type == "BOOL":
                return node.children[0].value == "True"

            if token == "list":
                results = []

                for item in node.children:
                    results.append(self.parse_tree(item))

                return results

            if token == "var":
                self.state[node.children[0].children[0].value] = self.parse_tree(
                    node.children[1]
                )
                self.state["last"] = self.state[node.children[0].children[0].value]
                continue

            if token.value == "if":
                # copy self.state
                last_state_before_if = copy.deepcopy(self.state)["last"]

                self.state["ctx"] = {
                    "if": True,
                }

                # if equality, check if equal

                statement = node.children[0]

                statement = self.parse_tree(statement)

                if statement is None:
                    continue

                if statement is False:
                    return

                self.state["last"] = last_state_before_if

            if token.value == "make":
                self.make(node.children)
                continue

            if token.value == None:
                continue

            if token.value == "run":
                function_name = node.children[0].value

                if function_name not in self.state["functions"]:
                    print(f"Function {function_name} does not exist.")
                    exit(1)

                function_args = self.state["functions"][function_name]

                for item in function_args:
                    self.parse_tree(item)

                continue

            if token.value == "literal":
                func = self.state["functions"][node.children[0].value]
            else:
                func = self.function_calls[token.value]

            if token.value == "negate" or token.value == "input":
                result = self.parse_tree(node.children[0])
                return func(result)

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
                        elif item.type == "input":
                            item.value = self.parse_tree(item.value)
                        elif item.type == "FLOAT":
                            item.value = float(item.value)

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

                    # print(context)

                    for item in context:
                        self.parse_tree(item)

                del self.state["ctx"]

                continue

            if len(node.children) == 1:
                if hasattr(node.children[0], "value"):
                    value = node.children[0].value
                else:
                    value = self.parse_tree(node.children[0])
            elif all([hasattr(item, "value") for item in node.children]):
                value = [item.value for item in node.children]
            else:
                value = [self.parse_tree(item) for item in node.children]

            if token.value == "literal":
                result = self.parse_tree(self.state["functions"][value])
            else:
                result = func(value)

            if result is not None:
                self.state["last"] = result
                self.state["output"] = {"text": result}

            self.state["last_function_type"] = token.value
            self.state["last_function_args"] = [value]

            if token.value == "load":
                self.state["image_stack"].append(result)


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
            handle_unexpected_characters(e, code.lstrip())
        except UnexpectedToken as e:
            handle_unexpected_token(e)

        session.parse_tree(tree)


@click.command()
@click.option("--validate", default=False, help="")
@click.option("--ref", default=False, help="Name of the file")
@click.option("--debug", default=False, help="To debug")
@click.option("--file", default=None, help="Name of the file")
@click.option("--repl", default=None, help="To enter to visionscript console")
@click.option("--notebook/--no-notebook", help="Start a notebook environment")
@click.option("--cloud/--no-cloud", help="Start a cloud deployment environment")
@click.option("--deploy", help="Deploy a .vic file", default=None)
@click.option(
    "--name",
    help="Application name (used if you are deploying your app via --deploy)",
    default=None,
)
@click.option(
    "--description",
    help="Application description (used if you are deploying your app via --deploy)",
    default=None,
)
@click.option("--api-key", help="API key for deploying your app", default=None)
@click.option(
    "--api-url",
    help="API url for deploying your app",
    default="http://localhost:6999/create",
)
def main(
    validate,
    ref,
    debug,
    file,
    repl,
    notebook,
    cloud,
    deploy,
    name,
    description,
    api_key,
    api_url,
) -> None:
    if validate:
        print("Script is a valid VisionScript program.")
        exit(0)

    if ref:
        print(USAGE.strip())

    if notebook:
        print("Starting notebook...")
        import webbrowser

        from visionscript.notebook import app

        # webbrowser.open("http://localhost:5001/notebook?" + str(uuid.uuid4()))

        app.run(debug=True, host="localhost", port=5001)

        return

    if cloud:
        print("Starting cloud deployment environment...")

        from visionscript.cloud import app

        app.run(debug=True, port=6999)

    if file is not None:
        with open(file, "r") as f:
            code = f.read() + "\n"

        tree = parser.parse(code.lstrip())

        if debug:
            print(tree.pretty())
            exit()

        session = VisionScript()

        if deploy:
            if not name or not description or not api_key or not api_url:
                print("Please provide a name, description, api key, and api url.")
                return

            session.notebook = True

            session.check_inputs(code)

            import requests

            app_slug = name.translate(
                str.maketrans("", "", string.punctuation.replace("-", ""))
            ).replace(" ", "-")

            deploy_request = requests.post(
                api_url,
                json={
                    "title": name,
                    "slug": app_slug,
                    "api_key": api_key,
                    "description": description,
                    "script": code,
                    "variables": session.state["input_variables"],
                },
            )

            if deploy_request.ok:
                print("App deployed to", deploy_request.json()["url"])

            return

        session.parse_tree(tree)

    if repl == "console":
        activate_console(parser)


if __name__ == "__main__":
    main()
