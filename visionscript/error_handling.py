from spellchecker import SpellChecker
import sys

from visionscript.constants import ERROR_CODE
from visionscript.usage import (
    language_grammar_reference,
    lowercase_language_grammar_reference,
)


def visionscript_exception_handler(_, exception, _1):
    print(ERROR_CODE, exception)


class PathNotExists(Exception):
    def __init__(self, path):
        sys.excepthook = visionscript_exception_handler
        self.path = path

    def __str__(self):
        return f"The file '{self.path}' does not exist."


class StackEmpty(Exception):
    def __init__(self, stack):
        sys.excepthook = visionscript_exception_handler
        self.stack = stack

    def __str__(self):
        if self.stack == "image_stack":
            return "You need to load an image before you can use the image stack."
        elif self.stack == "pose_stack":
            return "You need to run DetectPose[] before you can use the pose stack."


class SetFunctionError(Exception):
    def __init__(self, function):
        sys.excepthook = visionscript_exception_handler
        self.function = function

    def __str__(self):
        return f"The '{self.function}' model is not available."


class ImageOutOfBounds(Exception):
    def __init__(self, x, y):
        sys.excepthook = visionscript_exception_handler
        self.x = x
        self.y = y

    def __str__(self):
        return f"The image is out of bounds at ({self.x}, {self.y})."


class CameraNotAccessible(Exception):
    def __init__(self):
        sys.excepthook = visionscript_exception_handler

    def __str__(self):
        return "The camera is not accessible."


class ImageNotSupported(Exception):
    def __init__(self, image):
        sys.excepthook = visionscript_exception_handler
        self.image = image

    def __str__(self):
        return f"The image '{self.image}' is in an unsupported format. Supported formats are: .jpg, .jpeg, and .png"

class ImageCorrupted(Exception):
    def __init__(self, image):
        sys.excepthook = visionscript_exception_handler
        self.image = image

    def __str__(self):
        return f"The image '{self.image}' is corrupt and cannot be opened."

spell = SpellChecker()


def handle_unexpected_characters(e, code, interactive=False):
    # if line doesn't end with ], add it
    if not code.strip().endswith("]"):
        code += "]"

        return

    # if space between statement and [, remove it
    # get position of [
    position = code.find("[")

    if code[position - 1] == " ":
        code = code[: position - 1] + code[position:]

        return

    # replace all “ with "
    code = code.replace("“", '"')
    code = code.replace("”", '"')

    # raise error if character not in grammar
    if e.char not in ["[", "]", "'", '"', ",", " ", '"', '"', "\n", "\t", "\r"]:
        print(ERROR_CODE, f"Syntax error on line {e.line}, column {e.column}.")
        print(ERROR_CODE, f"Unexpected character: {e.char!r}")
        exit(1)

    # raise error if class doesn't exist
    line = e.line
    column = e.column

    # check if function name in grammar
    function_name = code.strip().split("\n")[line - 1].split("[")[0].strip()

    language_grammar_reference_keys = language_grammar_reference.keys()

    if function_name in language_grammar_reference_keys:
        print(ERROR_CODE, f"Syntax error on line {line}, column {column}.")
        print(ERROR_CODE, f"Unexpected character: {e.char!r}")
        exit(1)

    spell.known(lowercase_language_grammar_reference)
    spell.word_frequency.load_words(lowercase_language_grammar_reference)

    alternatives = spell.candidates(function_name)

    if len(alternatives) == 0:
        print(ERROR_CODE, f"Function {function_name} does not exist.")
        exit(1)

    print(
        ERROR_CODE,
        f"Function '{function_name}' does not exist. Did you mean one of these?",
    )
    print("-" * 10)

    for item in list(alternatives):
        if item.lower() in lowercase_language_grammar_reference:
            print(
                list(language_grammar_reference.keys())[
                    lowercase_language_grammar_reference.index(item.lower())
                ]
            )

    if interactive is False:
        exit(1)

    return


def handle_unexpected_token(e, interactive=False):
    line = e.line
    column = e.column

    print(ERROR_CODE, f"Syntax error on line {line}, column {column}.")
    print(f"Unexpected token: {e.token!r}")
    if interactive is False:
        exit(1)
