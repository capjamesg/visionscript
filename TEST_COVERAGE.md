# VisionScript Language Test Coverage

The VisionScript Language is the language used to write VisionScript programs. This document contains a list of all built-in VisionScript functions, and the state of their test coverage.

This document pertains only to the VisionScript Language and runtime, not Notebooks or Cloud.

This document does not say which tests are passing. It only says whether a function has tests.

To run the tests, run `pytest tests/` in the root directory of this repository.

## Manual Testing

Some methods need manual testing because they use a webcam. The following functions must be tested manually:

- [ ] `Breakpoint[]`
- [X] `Compare[]`
- [X] `GetFPS[]`
- [X] `Show[]`
- [X] `ShowText[]`
- [X] `UseCamera[]`

## Automated Testing

### Functions

- [X] `Blur[]`
- [X] `Break[]`
- [X] `Caption[]`
- [X] `Classify[]`
- [X] `ComparePose[]`
- [X] `Count[]`
- [ ] `CountInRegion[]`
- [ ] `Cutout[]`
- [X] `Describe[]`
- [X] `Detect[]`
- [X] `DetectPose[]`
- [X] `Exit[]`
- [ ] `FilterByClass[]`
- [X] `Find[]`
- [X] `First[]`
- [ ] `GetDistinctScenes[]`
- [X] `GetEdges[]`
- [X] `GetText[]`
- [X] `Greyscale[]`
- [X] `If[]`
- [X] `Import[]`
- [X] `In[]` (folder of images)
- [ ] `In[]` (video file)
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
- [ ] `Replace[]`
- [X] `Reset[]`
- [X] `Resize[]`
- [X] `Rotate[]`
- [X] `Save[]`
- [X] `Say[]`
- [ ] `Search[]`
- [X] `Segment[]`
- [ ] `Select[]`
- [X] `SetBrightness[]`
- [ ] `SetConfidence[]`
- [X] `Similarity[]`
- [X] `Size[]`
- [X] `Use[]`
- [X] `Profile[]`
- [ ] `Web[]`

### Language Features

- [X] Increment
- [X] Decrement
- [X] Comment