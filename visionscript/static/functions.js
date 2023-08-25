
const FUNCTIONS = {
    "Input": {
        "Load": {
            "args": ["file"],
            "description": "Load an image from a file",
            "example": "Load[\"./folder/abbey.jpg\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
    },
    "Process": {
        "Greyscale": {
            "args": [],
            "description": "Convert an image to greyscale",
            "example": "Greyscale[]",
            "supports_arguments": false
        },
        "Rotate": {
            "args": ["angle"],
            "description": "Rotate an image",
            "example": "Rotate[90]",
            "supports_arguments": true
        },
        "Resize": {
            "args": ["width", "height"],
            "description": "Resize an image",
            "example": "Resize[100, 100]",
            "supports_arguments": true
        },
        "Crop": {
            "args": ["x", "y", "width", "height"],
            "description": "Crop an image",
            "example": "Crop[0, 0, 100, 100]",
            "supports_arguments": true,
            "argument_default": "0, 0, 0, 0"
        },
        "Blur": {
            "args": [],
            "description": "Blur an image",
            "example": "Blur[]",
            "supports_arguments": false
        },
        "SetBrightness": {
            "args": ["amount"],
            "description": "Adjust the brightness of an image",
            "example": "SetBrightness[50]",
            "supports_arguments": true
        },
        "Replace": {
            "args": ["file"],
            "description": "Replace part of an image",
            "example": "Replace[\"red\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Cutout": {
            "args": ["x", "y", "width", "height"],
            "description": "Cut out part of an image",
            "example": "Cutout[0, 0, 100, 100]",
            "supports_arguments": true,
            "argument_default": "0, 0, 0, 0"
        },
        "Size": {
            "args": [],
            "description": "Get the size of an image",
            "example": "Size[]",
            "supports_arguments": false
        },
        "Contrast": {
            "args": ["amount"],
            "description": "Adjust the contrast of an image",
            "example": "Contrast[1.5]",
            "supports_arguments": true
        }
    },
    "Find": {
        "SetRegion": {
            "args": ["x", "y", "width", "height"],
            "description": "Set the region to search for objects in (use before Detect[] or Segment[])",
            "example": "SetRegion[0, 0, 100, 100]",
            "supports_arguments": true,
            "argument_default": "0, 0, 0, 0"
        },
        "FilterByClass": {
            "args": ["object"],
            "description": "Filter objects by class",
            "example": "FilterByClass[\"person\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Classify": {
            "args": ["object"],
            "description": "Classify an image",
            "example": "Classify[\"person\", \"cat\"]",
            "supports_arguments": true,
            "argument_default": "\"\", \"\""
        },
        "Detect": {
            "args": ["object"],
            "description": "Detect objects in an image",
            "example": "Detect[\"person\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "DetectPose": {
            "args": ["object"],
            "description": "Detect a pose in an image",
            "example": "DetectPose[]",
            "supports_arguments": false
        },
        "ComparePose": {
            "args": ["object"],
            "description": "Compare poses in two images",
            "example": "ComparePose[]",
            "supports_arguments": false
        },
        "Segment": {
            "args": ["object"],
            "description": "Segment objects in an image",
            "example": "Segment[\"person\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Search": {
            "args": ["file"],
            "description": "Build a search engine with loaded images",
            "example": "Search[\"./image.png\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Caption": {
            "args": [],
            "description": "Caption an image",
            "example": "Caption[]",
            "supports_arguments": false
        },
        "Count": {
            "args": ["object"],
            "description": "Count objects in an image",
            "example": "Count[\"person\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "ReadQR": {
            "args": [],
            "description": "Read a QR code in an image",
            "example": "ReadQR[]",
            "supports_arguments": false
        },
        "GetText": {
            "args": [],
            "description": "Get the text in an image",
            "example": "GetText[]",
            "supports_arguments": false
        },
        "Similarity": {
            "args": [],
            "description": "Find the similarity between two images",
            "example": "Similarity[]",
            "supports_arguments": false,
        },
        "GetColours": {
            "args": [],
            "description": "Get the most common colours in an image",
            "example": "GetColours[]",
            "supports_arguments": false
        },
        "GetEdges": {
            "args": [],
            "description": "Get the edges in an image",
            "example": "GetEdges[]",
            "supports_arguments": false
        }
    },
    "Output": {
        "Say": {
            "args": [],
            "description": "Output the result of the previous function",
            "example": "Say[]",
            "supports_arguments": false
        },
        "Show": {
            "args": [],
            "description": "Show the result of the previous function",
            "example": "Show[]",
            "supports_arguments": false
        },
        "Compare": {
            "args": ["file"],
            "description": "Compare two or more images",
            "example": "Compare[]",
            "supports_arguments": false
        },
        "Save": {
            "args": ["file"],
            "description": "Save the result of the previous function to a file",
            "example": "Save[\"./folder/abbey.jpg\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Read": {
            "args": ["file"],
            "description": "Read the last value from state",
            "example": "Read[]",
            "supports_arguments": false
        },
        "GetDistinctScenes": {
            "args": [],
            "description": "Get the distinct scenes in a video",
            "example": "GetDistinctScenes[]",
            "supports_arguments": false
        },
        "GetUniqueAppearances": {
            "args": [],
            "description": "Get the unique appearances of an object in a video",
            "example": "GetUniqueAppearances[\"person\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        }
    },
    "Logic": {
        "If": {
            "args": ["condition"],
            "description": "If a condition is true, run the next function",
            "example": "If[\"person\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "In": {
            "args": [],
            "description": "Iterate over a folder of images",
            "example": "In[\"./folder\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Use": {
            "args": ["file"],
            "description": "Specify a model for use",
            "example": "Use[\"groundingdino\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Web": {
            "args": ["url"],
            "description": "Make a web request",
            "example": "Web[\"https://example.com/turn-on-lights\"]",
            "supports_arguments": true,
            "argument_default": "\"\""
        },
        "Reset": {
            "args": [],
            "description": "Reset the state of the program",
            "example": "Reset[]",
            "supports_arguments": false
        }
    },
    "Deploy": {
        "Input": {
            "args": [],
            "description": "Specify a custom field users can input with a deployed model",
            "example": "Input[]",
            "supports_arguments": false
        },
    }
};