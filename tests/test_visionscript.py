import visionscript as lang
import warnings
import os
import pytest

from visionscript import error_handling

TEST_DIR = os.path.join(os.path.dirname(__file__), "vics/")
VALID_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "valid_output/")
RAISES_EXCEPTIONS_TEST_DIR = os.path.join(os.path.dirname(__file__), "vics/raises_exceptions/")

@pytest.mark.skip
def test_visionscript_program(file):
    session = lang.VisionScript()
    
    with open(TEST_DIR + file, "r") as f:
        print("Testing " + file)
        file_path = os.path.join(TEST_DIR, file)
        session.parse_tree(lang.parser.parse(f.read() + "\n"))
        return session.state["output"]["text"]

def test_path_not_exists():
    file = "raises_exceptions/path_not_exists.vic"

    with pytest.raises(error_handling.PathNotExists):
        test_visionscript_program(file)

def test_classify():
    file = "classify_image.vic"

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