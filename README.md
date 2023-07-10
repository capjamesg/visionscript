![A VisionScript Notebook counting people in an image](https://raw.githubusercontent.com/capjamesg/visionscript/main/notebook.png)

# VisionScript

[VisionScript](https://visionscript.dev) is an abstract programming language for doing common computer vision tasks, fast.

VisionScript is built in Python, offering a simple syntax for running object detection, classification, and segmentation models. [Read the documentation](https://visionscript.dev/docs/).

## Get Started 🚀

First, install VisionScript:

```bash
pip install visionscript
```

You can then run VisionScript using:

```bash
visionscript --repl
```

This will open a VisionScript REPL in which you can type commands.

### Run a File 📁

To run a VisionScript file, use:

```bash
visionscript ./your_file.vic
```

### Use VisionScript in a Notebook 📓

VisionScript offers an interactive web notebook through which you can run VisionScript code.

To use the notebook, run:

```bash
visionscript --notebook
```

This will open a notebook in your browser. Notebooks are ephermal. You will need to copy your code to a file to save it.

## Quickstart 🚀

### Find people in an image using object detection

```
Load["./photo.jpg"]
Detect["person"]
Say[]
```

### Find people in all images in a folder using object detection

```
In["./images"]
    Detect["person"]
    Say[]
```

### Replace people in a photo with an emoji

```
Load["./abbey.jpg"]
Size[]
Say[]
Detect["person"]
Replace["emoji.png"]
Save["./abbey2.jpg"]
```

### Classify an image

```
Load["./photo.jpg"]
Classify["apple", "banana"]
```

## Installation 👷

To install VisionScript, clone this repository and run `pip install -r requirements.txt`.

Then, make a file ending in `.vic` in which to write your VisionScript code.

When you have written your code, run:

```bash
python3 lang.py --file ./your_file.vic
```

### Run in debug mode

Running in debug mode shows the full Abstract Syntax Tree (AST) of your code.

```bash
python3 lang.py --file ./your_file.vic --debug
```

Debug mode is useful for debugging code while adding new features to the VisionScript language.

## Inspiration 🌟

The inspiration behind this project was to build a simple way of doing one-off tasks.

Consider a scenario where you want to run zero-shot classification on a folder of images. With VisionScript, you can do this in two lines of code:

```
In["./images"]
    Classify["cat", "dog"]
```

VisionScript is not meant to be a full programming language for all vision tasks, rather an abstract way of doing common tasks.

VisionScript is ideal if you are new to concepts like "classify" and "segment" and want to explore what they do to an image.

### Syntax

The syntax is inspired by both Python and the Wolfram Language. VisionScript is an interpreted language, run line-by-line like Python. Statements use the format:

```
Statement[argument1, argument2, ...]
```

This is the same format as the Wolfram Language.

### Lexical Inference and Memory

An (I think!) unique feature in VisionScript compared to other languages is lexical inference.

You don't need to declare variables to store images, etc. Rather, you can let VisionScript do the work. Consider this example:

```
Load["./photo.jpg"]
Size[]
Say[]
```

Here, `Size[]` and `Say[]` do not have any arguments. Rather, they use the last input. Wolfram Alpha has a feature to get the last input using `%`. VisionScript uses the same concept, but with a twist.

Indeed, `Size[]` and `Say[]` don't accept any arguments.

## Developer Setup 🛠

If you want to add new features or fix bugs in the VisionScript language, you will need to set up a developer environment.

To do so, clone the language repository:

```bash
git clone https://github.com/capjamesg/VisionScript
```

Then, install the required dependencies and VisionScript:

```bash
pip install -r requirements.txt
pip install -e .
```

Now, you can run VisionScript using:

```bash
python3 lang.py
```

### Tests

Tests are run to ensure programs execute in full. Tests do not verify the output of each statement, although this will be added.

For now, you can run all test cases using the following command:

```bash
python3 test.py
```

## Supported Models 📚

VisionScript provides abstract wrappers around:

- [CLIP](https://github.com/openai/clip) by OpenAI (Classification)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Object Detection Training, Segmentation Training)
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) by CASIA-IVA-Lab. (Segmentation)
- [GroundedSAM](https://docs.autodistill.com/base_models/groundedsam/) (Object Detection, Segmentation)
- [BLIP](https://github.com/salesforce/BLIP) (Caption Generation)
- [ViT](https://github.com/autodistill/autodistill-vit) (Classification Training)

## License 📝

This project is licensed under an [MIT license](LICENSE).