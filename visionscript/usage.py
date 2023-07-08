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
    "Label": "Label images",
    "Get": "Get the value of a variable",
    "Read": "Read a value",
    "Compare": "Compare two or more images",
    "Cutout": "Cutout the last detections",
    "Count": "Count the last detections",
    "GetColors": "Get the colors in the image",
    "GetColours": "Get the colours in the image",
    "GetText": "Get the text in the image",
    "Greyscale": "Convert the image to greyscale",
    "Select": "Select the last image",
    "Paste": "Paste the last image in a specified position",
    "PasteRandom": "Paste the last image in a random location",
    "Resize": "Resize the last image",
    "Blur": "Blur the last image",
    "SetBrightness": "Set the brightness of the last image",
    "Search": "Search for an image",
    "Similarity": "Get the similarity of two images",
    "ReadQR": "Read a QR code",
    "Reset": "Reset the state",
    "Not": "Negate an expression",
    "Is it a": "Check if the image is a certain object",
    "Find": "Find an object in the image",
    "Describe": "Describe the image",
    "Import": "Import a module",
    "Rotate": "Rotate the image",
    "Exit": "Exit the program",
    "Make": "Make a function",
}

lowercase_language_grammar_reference = [
    item.lower() for item in language_grammar_reference
]

USAGE = """
VisionScript (VIC) is a visual programming language for computer vision.

VisionScript is a line-based language. Each line is a function call.

View the full documentation at:

https://visionscript.dev/docs
"""
