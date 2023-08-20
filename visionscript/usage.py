import json
import os

with open(os.path.join(os.path.dirname(__file__), "reference.json")) as f:
    language_grammar_reference = json.load(f)

lowercase_language_grammar_reference = [
    item.lower() for item in language_grammar_reference
]

USAGE = """
VisionScript (VIC) is a visual programming language for computer vision.

VisionScript is a line-based language. Each line is a function call.

View the full documentation at:

https://visionscript.dev/docs
"""
