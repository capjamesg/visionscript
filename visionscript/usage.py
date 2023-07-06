language_grammar_reference = {
    "Load": "Load an image",
    "Save": "Save an image",
    "Size": "Get the size of the image (width, height)",
    "Say": "Say the result of the last function",
    "Detect": "Find objects in the image",
    "Replace": "Replace the last detections with a random image",
    "Cutout": "Cutout the last detections",
    "Count": "Count the last detections",
    "Segment": "Segment the image",
    "CountInRegion": "Count the last detections in the region (x1, y1, x2, y2)",
    "Classify": "Classify the image in the provided categories",
    "Show": "Show the image",
    "In": "Iterate over the files in a directory",
    "If": "If statement",
    "Train": "Train a model",
}

lowercase_language_grammar_reference = [
    item.lower() for item in language_grammar_reference
]

USAGE = """
VisionScript (VIC) is a visual programming language for computer vision.

VisionScript is a line-based language. Each line is a function call.

Language Reference
------------------
Load["./abbey.jpg"] -> Load the image
Size[] -> Get the size of the image
Say[] -> Say the result of the last function
Detect["person"] -> Detect the person
Replace["emoji.png"] -> Replace the person with black image
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