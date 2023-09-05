import visionscript as lang
import os
import pytest
from PIL import Image
import supervision as sv

from visionscript.state import init_state
from visionscript import error_handling

TEST_DIR = os.path.join(os.path.dirname(__file__), "vics")
VALID_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "valid_output/")
RAISES_EXCEPTIONS_TEST_DIR = os.path.join(
    os.path.dirname(__file__), "vics/raises_exceptions/"
)


@pytest.mark.skip
def test_visionscript_program(file, return_raw_object=False, input_variables={}):
    session = lang.VisionScript()

    if input_variables:
        session.state["input_variables"] = {
            **session.state["input_variables"],
            **input_variables,
        }

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


def test_comment():
    file = "comment.vic"

    assert test_visionscript_program(file) == "banana"


def test_import():
    file = "import.vic"

    assert test_visionscript_program(file) == "banana"


def test_find_in_images():
    file = "find_in_images.vic"

    assert (
        test_visionscript_program(file)
        == open(os.path.join(VALID_OUTPUT_DIR, "find_in_images.vic.txt"), "r").read()
    )


def test_load_detect_save():
    file = "load_detect_save.vic"

    assert test_visionscript_program(file) == "Saved to ./bus1.jpg"


def test_similarity():
    file = "similarity.vic"

    assert test_visionscript_program(file) == [0]


def test_count():
    file = "count.vic"

    assert test_visionscript_program(file) == 4


def test_count_in_region():
    file = "count_in_region.vic"

    assert test_visionscript_program(file) == 4


def test_filter_by_class():
    file = "filter_by_class.vic"

    assert test_visionscript_program(file) == 1


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

    assert (
        test_visionscript_program(file, True).state["run_video_in_background"] == True
    )


def test_use_model():
    file = "use.vic"

    assert (
        test_visionscript_program(file, True).state["current_active_model"]
        == "grounding_dino"
    )


def test_use_roboflow_model():
    file = "use_roboflow.vic"

    assert (
        test_visionscript_program(file, True).state["current_active_model"]
        == "roboflow rock paper scissors"
    )


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

    assert (
        program_state["current_active_model"] == initial_state["current_active_model"]
    )


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


def test_replace():
    file = "replace.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/replace_with_color.jpg")
    reference = os.path.join(__file__, "valid_output/replace_with_color.jpg")

    assert compare_two_images_for_equality(used_file, reference)


def test_web():
    file = "web.vic"

    with open(os.path.join(os.path.dirname(__file__), "output/web.html"), "w") as f:
        f.write(test_visionscript_program(file))

    assert (
        test_visionscript_program(file)
        == open(os.path.join(VALID_OUTPUT_DIR, "web.html"), "r").read()
    )


def test_paste():
    file = "paste.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/paste.jpg")
    reference = os.path.join(__file__, "valid_output/paste.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_blur():
    file = "rotate.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "valid_output/rotate.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_profile():
    file = "profile.vic"

    assert "Total run time:" in test_visionscript_program(file)


def test_size():
    file = "size.vic"

    assert test_visionscript_program(file) == (1080, 810)


def not_true_or_false():
    file = "not.vic"

    assert test_visionscript_program(file) == False


def in_video():
    file = "in_video.vic"

    assert test_visionscript_program(file) == 240


def read_qr_code():
    file = "readqr.vic"

    assert test_visionscript_program(file) == "https://jamesg.blog"


def test_detect_pose():
    file = "detect_pose.vic"

    assert (
        test_visionscript_program(file)
        == open(os.path.join(VALID_OUTPUT_DIR, "detect_pose.vic.txt"), "r").read()
    )


def test_compare_pose():
    file = "compare_pose.vic"

    assert (
        test_visionscript_program(file)
        == open(os.path.join(VALID_OUTPUT_DIR, "compare_pose.vic.txt"), "r").read()
    )


def test_save():
    file = "save.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "output/bus_cutout_saved.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_replace_in_images():
    file = "replace_in_images.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/replace_in_images.jpg")
    reference = os.path.join(__file__, "valid_output/replace_in_images.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_replace_with_color():
    file = "replace_with_color.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/replace_with_color.jpg")
    reference = os.path.join(__file__, "valid_output/replace_with_color.jpg")

    assert compare_two_images_for_equality(used_file, reference)


def test_cutout():
    file = "cutout.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "output/bus_cutout.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_resize():
    file = "resize.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "output/bus_resized.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_get_edges():
    file = "get_edges.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "output/bus_edges.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_set_brightness():
    file = "set_brightness.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    reference = os.path.join(__file__, "output/bus_brightness.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_if():
    file = "if.vic"

    assert test_visionscript_program(file) == "More than two people!"


def test_read():
    file = "read.vic"

    assert test_visionscript_program(file) == "More than two people!"


def test_get_text():
    file = "get_text.vic"

    assert test_visionscript_program(file) == "Raft Consensus Algorithm"


def test_exit():
    file = "exit.vic"

    with pytest.raises(SystemExit):
        test_visionscript_program(file)


def test_increment():
    file = "increment.vic"

    num_files_in_dir = len(
        os.listdir(os.path.join(os.path.dirname(__file__), "images/"))
    )

    assert test_visionscript_program(file) == num_files_in_dir


def test_decrement():
    file = "decrement.vic"

    num_files_in_dir = len(
        os.listdir(os.path.join(os.path.dirname(__file__), "images/"))
    )

    # negative (-) value because the script is counting down
    assert test_visionscript_program(file) == -num_files_in_dir


def test_input():
    file = "input.vic"

    assert (
        test_visionscript_program(
            file, input_variables={"file": "./tests/images/bus.jpg"}
        )
        == 4
    )


def test_setconfidence():
    file = "setconfidence.vic"

    assert test_visionscript_program(file) == 1


def test_select():
    file = "select.vic"

    result = test_visionscript_program(file)

    assert (
        isinstance(result, sv.Detections) and len(result) == 1 and len(result.xyxy) > 0
    )


def test_search():
    file = "search.vic"

    bus_image = os.path.join(os.path.dirname(__file__), "images/bus.jpg")
    bus_image_as_pil = Image.open(bus_image)

    assert (
        test_visionscript_program(file, True).state["image_stack"][-1]
        == bus_image_as_pil
    )


def get_distinct_scenes():
    file = "get_distinct_scenes.vic"

    with open(
        os.path.join(os.path.dirname(__file__), "output/get_distinct_scenes.txt"), "w"
    ) as f:
        f.write(test_visionscript_program(file))

    assert (
        test_visionscript_program(file)
        == open(os.path.join(VALID_OUTPUT_DIR, "get_distinct_scenes.txt"), "r").read()
    )


def say():
    file = "say.vic"

    with open(os.path.join(os.path.dirname(__file__), "output/say.txt"), "w") as f:
        f.write(test_visionscript_program(file))

    assert (
        test_visionscript_program(file)
        == open(os.path.join(VALID_OUTPUT_DIR, "say.txt"), "r").read()
    )


def variable_assignment():
    file = "variable_assignment.vic"

    assert test_visionscript_program(file) == 6
