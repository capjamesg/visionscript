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
- [ ] `Get[]`
- [ ] `Set[]`
- [ ] `Remove[]`
- [ ] `Wait[]`
- [ ] `Track[]`
- [ ] `Remove[]`
- [ ] `Apply[]`
- [ ] `Grid[]`
- [ ] `Shuffle[]`
- [ ] `GetColors[]`
- [ ] `GetColours[]`
- [ ] `IsItA[]`
- [ ] `Is[]`
- [ ] `Merge[]`
- [ ] `Say[]`

### Language Features

- [X] Increment
- [X] Decrement
- [X] Comment
- [X] Assignment
- [ ] List
- [ ] Associative array
- [ ] Greater than
- [ ] Less than
- [ ] Greater than or equal to
- [ ] Less than or equal to
- [ ] Equal to
- [ ] Not equal to

### Exceptions

- [X] `visionscript.errors.PathNotExists`