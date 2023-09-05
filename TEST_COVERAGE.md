# VisionScript Language Test Coverage

The VisionScript Language is the language used to write VisionScript programs. This document contains a list of all built-in VisionScript functions, and the state of their test coverage.

This document pertains only to the VisionScript Language and runtime, not Notebooks or Cloud.

This document does not say which tests are passing. It only says whether a function has tests.

To run the tests, run `pytest tests/` in the root directory of this repository.

## Manual Testing

Some methods need manual testing because they use a webcam. The following functions must be tested manually:

- [X] `Breakpoint[]`
- [X] `Compare[]`
- [X] `GetFPS[]`
- [X] `Show[]`
- [X] `ShowText[]`
- [X] `UseCamera[]`
- [ ] `Deploy[]`

## Automated Testing

A [1] indicates a test `.vic` file has been written but the corresponding Python test has not been added to `tests/test_visionscript.py`.

### Functions

- [X] `Blur[]`
- [X] `Break[]`
- [X] `Caption[]`
- [X] `Classify[]`
- [X] `ComparePose[]`
- [X] `Count[]`
- [X] `CountInRegion[]`
- [X] `Cutout[]`
- [X] `Describe[]`
- [X] `Detect[]`
- [X] `DetectPose[]`
- [X] `Exit[]`
- [X] `FilterByClass[]`
- [X] `Find[]`
- [X] `First[]`
- [x] `GetDistinctScenes[]`
- [X] `GetEdges[]`
- [X] `GetText[]`
- [X] `Greyscale[]`
- [X] `If[]`
- [X] `Import[]`
- [X] `In[]` (folder of images)
- [X] `In[]` (video file)
- [X] `Input[]`
- [X] `Last[]`
- [ ] `Load[]`
- [X] `Make[]`
- [X] `Not[]`
- [ ] `Paste[]`
- [ ] `PasteRandom[]`
- [X] `Random[]`
- [X] `Read[]`
- [X] `ReadQR[]`
- [X] `Replace[]`
- [X] `Reset[]`
- [X] `Resize[]`
- [X] `Rotate[]`
- [X] `Save[]`
- [X] `Say[]`
- [X] `Search[]`
- [X] `Segment[]`
- [X] `Select[]`
- [X] `SetBrightness[]`
- [X] `SetConfidence[]`
- [X] `Similarity[]`
- [X] `Size[]`
- [X] `Use[]`
- [X] `Profile[]`
- [X] `Web[]`
- [ ] `Crop[]`
- [ ] `Contains[]`
- [X] `Get[]`
- [X] `Set[]`
- [ ] `Remove[]` [1]
- [X] `Wait[]`
- [ ] `Track[]`
- [ ] `GetUniqueAppearances[]`
- [ ] `Apply[]`
- [X] `Grid[]`
- [ ] `Shuffle[]`
- [X] `GetColors[]`
- [X] `GetColours[]`
- [X] `IsItA[]`
- [X] `Is[]`
- [ ] `Merge[]`
- [ ] `Say[]` [1]

### Language Features

- [X] Increment
- [X] Decrement
- [X] Comment
- [X] Assignment
- [X] List
- [ ] Associative array [1]
- [X] Greater than
- [X] Less than
- [X] Greater than or equal to
- [X] Less than or equal to
- [X] Equal to
- [X] Not equal to

### Exceptions

- [X] `visionscript.errors.PathNotExists`
- [ ] `visionscript.errors.StackEmpty`
- [ ] `visionscript.errors.SetFunctionError`
- [ ] `visionscript.errors.ImageOutOfBounds`
- [ ] `visionscript.errors.CameraNotAccessible`

### States

- [ ] Ensure a buffer overflow does not occur when loading more than 1000 large images into memory
- [X] Ensure the `image_stack` never exceeds 100 images

## Models

- [ ] YOLOv8 Object Detection (small) [1]
- [ ] FastSAM [1]
- [ ] Grounding DINO [1]
- [ ] YOLOv8 Pose (small) [1]
- [ ] Roboflow `rock paper scissors` [1]