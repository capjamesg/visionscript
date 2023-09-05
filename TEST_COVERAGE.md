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

A [1] indicates a test `.vic` file has been written but the corresponding Python test has not been added to `tests/test_visionscript.py`
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
- [X] `Load[]`
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
- [ ] `Get[]` [1]
- [ ] `Set[]` [1]
- [ ] `Remove[]` [1]
- [ ] `Wait[]` [1]
- [ ] `Track[]`
- [ ] `GetUniqueAppearances[]`
- [ ] `Apply[]`
- [ ] `Grid[]`
- [ ] `Shuffle[]`
- [ ] `GetColors[]` [1]
- [ ] `GetColours[]` [1]
- [ ] `IsItA[]`
- [ ] `Is[]` [1]
- [ ] `Merge[]`
- [ ] `Say[]` [1]

### Language Features

- [X] Increment
- [X] Decrement
- [X] Comment
- [X] Assignment
- [ ] List [1]
- [ ] Associative array [1]
- [ ] Greater than [1]
- [ ] Less than [1]
- [ ] Greater than or equal to
- [ ] Less than or equal to
- [ ] Equal to [1]
- [ ] Not equal to [1]

### Exceptions

- [X] `visionscript.errors.PathNotExists`
- [ ] `visionscript.errors.StackEmpty`
- [ ] `visionscript.errors.SetFunctionError`
- [ ] `visionscript.errors.ImageOutOfBounds`
- [ ] `visionscript.errors.CameraNotAccessible`

### States

- [ ] Ensure a buffer overflow does not occur when loading more than 1000 large images into memory
- [ ] Ensure the `image_stack` never exceeds 100 images