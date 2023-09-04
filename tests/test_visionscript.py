import visionscript as lang
import warnings
import os
import pytest

from visionscript.state import init_state
from visionscript import error_handling

TEST_DIR = os.path.join(os.path.dirname(__file__), "vics")
VALID_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "valid_output/")
RAISES_EXCEPTIONS_TEST_DIR = os.path.join(os.path.dirname(__file__), "vics/raises_exceptions/")

@pytest.mark.skip
def test_visionscript_program(file, return_raw_object = False):
    session = lang.VisionScript()
    
    file_path = os.path.join(TEST_DIR, file)
    
    with open(file_path, "r") as f:
        session.parse_tree(lang.parser.parse(f.read() + "\n"))
        if return_raw_object is False:
            return session.state["output"]["text"]
        else:
            return session

def test_path_not_exists():
    file = "raises_exceptions/path_not_exists.vic"

    with pytest.raises(error_handling.PathNotExists):
        test_visionscript_program(file)

def test_classify():
    file = "classify_image.vic"

    assert test_visionscript_program(file) == "banana"

def test_import():
    file = "import.vic"

    assert test_visionscript_program(file) == "banana"

def test_find_in_images():
    file = "find_in_images.vic"

    assert test_visionscript_program(file) == open(os.path.join(VALID_OUTPUT_DIR, "find_in_images.vic.txt"), "r").read()

def test_load_detect_save():
    file = "load_detect_save.vic"

    assert test_visionscript_program(file) == "Saved to ./bus1.jpg"

def test_similarity():
    file = "similarity.vic"

    assert test_visionscript_program(file) == [0]

def test_count():
    file = "count.vic"

    assert test_visionscript_program(file) == 4

def test_random():
    file = "random.vic"

    options = [1, 2, 3]

    results = []

    for i in range(0, 20):
        result = test_visionscript_program(file) in options
        results.append(result)

    # make sure there is one of each possible result
    # technically this may fail by chance (if out of 20 tries, no instance of a single number appears), but it's unlikely
    for option in options:
        assert option in results

def test_use_background():
    file = "use_background.vic"

    assert test_visionscript_program(file, True).state["run_video_in_background"] == True

def test_use_model():
    file = "use.vic"

    assert test_visionscript_program(file, True).state["current_active_model"] == "grounding_dino"

def test_use_roboflow_model():
    file = "use_roboflow.vic"

    assert test_visionscript_program(file, True).state["current_active_model"] == "roboflow rock paper scissors"

def test_caption():
    file = "caption.vic"

    assert test_visionscript_program(file) == "a group of people walking down a street"

def test_describe():
    file = "caption.vic"

    assert test_visionscript_program(file) == "a group of people walking down a street"

def test_first():
    file = "first.vic"

    assert test_visionscript_program(file) == 1

def test_last():
    file = "last.vic"

    assert test_visionscript_program(file) == 3

def test_reset():
    initial_state = init_state()

    program_state = test_visionscript_program("reset.vic", True).state

    assert program_state["current_active_model"] == initial_state["current_active_model"]

@pytest.mark.skip
def compare_two_images_for_equality(image1, image2):
    from PIL import Image

    im1 = Image.open(image1)
    im2 = Image.open(image2)

    return im1 == im2

def test_blur():
    file = "blur.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "valid_output/blur.png")

    assert compare_two_images_for_equality(used_file, reference)

def test_greyscale():
    file = "greyscale.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "valid_output/greyscale.png")

    assert compare_two_images_for_equality(used_file, reference)

def test_size():
    file = "size.vic"

    assert test_visionscript_program(file) == (800, 600)

def not_true_or_false():
    file = "not.vic"

    assert test_visionscript_program(file) == False