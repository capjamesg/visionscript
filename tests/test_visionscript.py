import visionscript as lang
import os
import pytest
from PIL import Image
import supervision as sv
import time

from visionscript.state import init_state
from visionscript import error_handling
import numpy as np

TEST_DIR = os.path.join(os.path.dirname(__file__), "vics")
VALID_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "valid_output/")
RAISES_EXCEPTIONS_TEST_DIR = os.path.join(
    os.path.dirname(__file__), "vics/raises_exceptions/"
)


@pytest.mark.skip
def test_visionscript_program(
    file, return_raw_object=False, input_variables={}, session=None
):
    if not session:
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


def test_undefined_variable_or_function():
    file = "raises_exceptions/undefined_variable_or_function.vic"

    with pytest.raises(error_handling.UndefinedVariableOrFunction):
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

    assert len(test_visionscript_program(file, True).state["detections_stack"]) > 1


def test_load_image():
    file = "load_image.vic"

    assert len(test_visionscript_program(file, True).state["load_queue"]) == 1


def test_load_video():
    file = "load_video.vic"

    assert len(test_visionscript_program(file, True).state["load_queue"]) == 1


def test_load_folder():
    file = "load_folder.vic"

    assert len(test_visionscript_program(file, True).state["load_queue"]) == 3


def test_buffer_overload_prevention():
    file = "buffer_overload_prevention.vic"

    session = lang.VisionScript()

    for i in range(0, 101):
        test_visionscript_program(file, session=session)

    assert len(session.state["load_queue"]) == 100


def test_load_detect_save():
    file = "load_detect_save.vic"

    assert test_visionscript_program(file) == "Saved to ./bus1.jpg"


def test_similarity():
    file = "similarity.vic"

    assert round(test_visionscript_program(file), 1) == 1.0


def test_count():
    file = "count.vic"

    assert test_visionscript_program(file) == 4


def test_count_in_region():
    file = "count_in_region.vic"

    assert test_visionscript_program(file) == 1


def test_filter_by_class():
    file = "filter_by_class.vic"

    assert test_visionscript_program(file) == 6


def test_random():
    file = "random.vic"

    options = [1, 2, 3]

    results = []

    for i in range(0, 20):
        result = test_visionscript_program(file)
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
        == "groundingdino"
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

    used_file = os.path.join(os.path.dirname(__file__), "output/blur.jpg")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/blur.jpg")

    assert compare_two_images_for_equality(used_file, reference)


def test_grid():
    file = "grid.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "valid_output/grid.jpg")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/grid.jpg")

    assert compare_two_images_for_equality(used_file, reference)


def test_set_function_error():
    file = "raises_exceptions/set_function_error.vic"

    with pytest.raises(error_handling.SetFunctionError):
        test_visionscript_program(file)


def test_image_out_of_bounds():
    file = "raises_exceptions/image_out_of_bounds.vic"

    with pytest.raises(error_handling.ImageOutOfBounds):
        test_visionscript_program(file)


def test_image_not_supported():
    file = "raises_exceptions/image_not_supported.vic"

    with pytest.raises(error_handling.ImageNotSupported):
        test_visionscript_program(file)


def test_stack_empty():
    file = "raises_exceptions/stack_empty.vic"

    with pytest.raises(error_handling.StackEmpty):
        test_visionscript_program(file)


def test_image_corrupted():
    file = "raises_exceptions/image_corrupted.vic"

    with pytest.raises(error_handling.ImageCorrupted):
        test_visionscript_program(file)


def test_greyscale():
    file = "greyscale.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/greyscale.jpg")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/greyscale.jpg")

    assert compare_two_images_for_equality(used_file, reference)


def test_replace():
    file = "replace_with_color.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/replace_with_color.jpg")
    reference = os.path.join(
        os.path.dirname(__file__), "valid_output/replace_with_color.jpg"
    )

    assert compare_two_images_for_equality(used_file, reference)


def test_web():
    file = "web.vic"

    assert (
        test_visionscript_program(file)
        == open(os.path.join(os.path.dirname(__file__), "output/web.html"), "r").read()
    )


def test_paste():
    file = "paste.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/paste.jpg")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/paste.png")

    assert compare_two_images_for_equality(used_file, reference)


def test_rotate():
    file = "rotate.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/rotate.jpg")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/rotate.jpg")

    assert compare_two_images_for_equality(used_file, reference)


def test_profile():
    file = "profile.vic"

    assert (
        test_visionscript_program(file, True).state["ctx"].get("profile", False) == True
    )


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

    assert len(test_visionscript_program(file, True).state["poses_stack"]) == 1


def test_compare_pose():
    file = "compare_pose.vic"

    assert test_visionscript_program(file) == 1.0


def test_save():
    file = "save.vic"

    result = test_visionscript_program(file, True).state["image_stack"][-1]

    used_file = os.path.join(
        os.path.dirname(__file__), "valid_output/bus_cutout_saved.png"
    )
    reference = os.path.join(
        os.path.dirname(__file__), "valid_output/bus_cutout_saved.png"
    )

    # save result as PIL
    with open(used_file, "wb") as f:
        Image.fromarray(result).save(f)

    assert compare_two_images_for_equality(used_file, reference)


def test_save_to_csv():
    file = "save_to_csv.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/predictions.csv")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/predictions.csv")

    assert open(used_file, "r").read() == open(reference, "r").read()


def test_save_to_json():
    file = "save_to_json.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/predictions.json")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/predictions.json")

    # compare two jsons
    assert open(used_file, "r").read() == open(reference, "r").read()


def test_replace_in_images():
    file = "replace_in_images.vic"

    result = test_visionscript_program(file, True).state["image_stack"][-1]

    used_file = os.path.join(
        os.path.dirname(__file__), "valid_output/replace_in_images.png"
    )
    reference = os.path.join(
        os.path.dirname(__file__), "valid_output/replace_in_images.png"
    )

    with open(reference, "wb") as f:
        Image.fromarray(result).save(f)

    assert compare_two_images_for_equality(used_file, reference)


def test_replace_with_color():
    file = "replace_with_color.vic"

    result = test_visionscript_program(file, True).state["image_stack"][-1]

    used_file = os.path.join(
        os.path.dirname(__file__), "valid_output/replace_with_color.jpg"
    )
    reference = os.path.join(
        os.path.dirname(__file__), "valid_output/replace_with_color.jpg"
    )

    with open(reference, "wb") as f:
        Image.fromarray(result).save(f)

    assert compare_two_images_for_equality(used_file, reference)


def test_cutout():
    file = "cutout.vic"

    result = test_visionscript_program(file, True).state["image_stack"][-1]

    used_file = os.path.join(os.path.dirname(__file__), "valid_output/bus_cutout.png")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/bus_cutout.png")

    with open(reference, "wb") as f:
        Image.fromarray(result).save(f)

    assert compare_two_images_for_equality(used_file, reference)


def test_resize():
    file = "resize.vic"

    result = test_visionscript_program(file, True).state["image_stack"][-1]

    used_file = os.path.join(os.path.dirname(__file__), "valid_output/bus_resized.jpg")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/bus_resized.jpg")

    # save result as PIL
    with open(used_file, "wb") as f:
        Image.fromarray(result).save(f)

    assert compare_two_images_for_equality(used_file, reference)


def test_get_edges():
    file = "getedges.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/bus_edges.jpg")
    reference = os.path.join(os.path.dirname(__file__), "valid_output/bus_edges.jpg")

    assert compare_two_images_for_equality(used_file, reference)


def test_set_brightness():
    file = "setbrightness.vic"

    test_visionscript_program(file)

    used_file = os.path.join(os.path.dirname(__file__), "output/bus_brightness.jpg")
    reference = os.path.join(
        os.path.dirname(__file__), "valid_output/bus_brightness.jpg"
    )

    assert compare_two_images_for_equality(used_file, reference)


def test_if():
    file = "if.vic"

    assert test_visionscript_program(file) == "More than two people!"


def test_read():
    file = "read.vic"

    assert test_visionscript_program(file) == "More than two people!"


def test_get_text():
    file = "get_text.vic"

    assert test_visionscript_program(file) == "The Raft Consensus Algorithm"


def test_exit():
    file = "exit.vic"

    with pytest.raises(SystemExit):
        test_visionscript_program(file)


def test_increment():
    file = "increment.vic"

    num_files_in_dir = len(
        os.listdir(os.path.join(os.path.dirname(__file__), "directory_list_test/"))
    )

    assert test_visionscript_program(file) == num_files_in_dir


def test_decrement():
    file = "decrement.vic"

    num_files_in_dir = len(
        os.listdir(os.path.join(os.path.dirname(__file__), "directory_list_test/"))
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


def test_is():
    file = "is.vic"

    assert isinstance(test_visionscript_program(file), sv.Detections)


def test_wait():
    file = "wait.vic"

    start_time = time.time()

    test_visionscript_program(file)

    end_time = time.time()

    assert end_time - start_time >= 1


def get_colors():
    file = "getcolors.vic"

    assert test_visionscript_program(file) == "grey"


def get_colours():
    file = "getcolours.vic"

    assert test_visionscript_program(file) == "grey"


def define_list():
    file = "list.vic"

    assert test_visionscript_program(file) == 1


def test_get():
    file = "get.vic"

    assert test_visionscript_program(file) == 4


def test_set():
    file = "set.vic"

    assert test_visionscript_program(file) == 4


def test_greater_than():
    file = "greater_than.vic"

    assert test_visionscript_program(file) == "x is greater than 1"


def test_less_than():
    file = "less_than.vic"

    assert test_visionscript_program(file) == "x is less than 1"


def test_greater_than_or_equal_to():
    file = "greater_than_or_equal_to.vic"

    assert test_visionscript_program(file) == "x is greater than or equal to 1"


def test_less_than_or_equal_to():
    file = "less_than_or_equal_to.vic"

    assert test_visionscript_program(file) == "x is less than or equal to 1"


def test_equal_to():
    file = "equal_to.vic"

    assert test_visionscript_program(file) == "x is equal to 1"


def test_not_equal_to():
    file = "not_equal_to.vic"

    assert test_visionscript_program(file) == "x is not equal to 1"


def test_fastsam():
    file = "models/fastsam.vic"

    assert test_visionscript_program(file) == 1


def test_groundingdino():
    file = "models/groundingdino.vic"

    assert test_visionscript_program(file) == 1


def test_yolov8():
    file = "models/yolov8.vic"

    assert test_visionscript_program(file) == 1


def test_roboflow():
    file = "models/roboflow.vic"

    assert test_visionscript_program(file) == 1


def test_yolov8s_pose():
    file = "models/yolov8s-pose.vic"

    assert len(test_visionscript_program(file, True).state["poses_stack"]) == 1


def test_empty_file():
    file = "empty_file.vic"

    # we are checking that the program does not raise an error, not for any
    # particular output
    assert test_visionscript_program(file)


@pytest.mark.skip
def _test_sorted(items):
    """
    Test whether a list is sorted.
    """
    for i in range(0, len(items) - 1):
        if items[i] < items[i + 1]:
            raise Exception("List is not sorted")


@pytest.mark.skip
def _test_all_elements_in_list(unsorted, sorted):
    assert np.all(np.isin(unsorted, sorted)) == True


@pytest.mark.skip
def _check_detection_index_integrity(unsorted, sorted):
    # make sure all bboxes preserve sort order
    highest_unsorted_idx = unsorted.confidence.tolist().index(max(unsorted.confidence))
    highest_sorted_idx = sorted.confidence.tolist().index(max(sorted.confidence))

    # check that the bboxes are the same
    if not np.all(
        unsorted.xyxy[highest_unsorted_idx] == sorted.xyxy[highest_sorted_idx]
    ):
        assert False


def test_detections_by_confidence_ordering():
    """
    Test whether detections are ordered by confidence.
    """
    results = test_visionscript_program("detections_by_confidence_ordering.vic", True)

    if len(results.state["detections_stack"]) == 0:
        raise Exception("No detections found")

    sorted_detections = results.state["detections_stack"][-1]
    unsorted_detections = results.state["raw_detections_stack"][-1]

    _test_sorted(sorted_detections.confidence)

    _test_all_elements_in_list(unsorted_detections.xyxy, sorted_detections.xyxy)
    _test_all_elements_in_list(unsorted_detections.mask, sorted_detections.mask)
    _test_all_elements_in_list(
        unsorted_detections.confidence, sorted_detections.confidence
    )
    _test_all_elements_in_list(unsorted_detections.class_id, sorted_detections.class_id)
    _test_all_elements_in_list(
        unsorted_detections.tracker_id, sorted_detections.tracker_id
    )

    _check_detection_index_integrity(unsorted_detections, sorted_detections)
