# VisionScript Language Test Coverage

The VisionScript Language is the language used to write VisionScript programs. This document contains a list of all built-in VisionScript functions, and the state of their test coverage.

This document pertains only to the VisionScript Language and runtime, not Notebooks or Cloud.

This document does not say which tests are passing. It only says whether a function has tests.

To run the tests, run `pytest tests/` in the root directory of this repository.

## Functions

- [X] `Blur[]`
- [ ] `Break[]`
- [ ] `Breakpoint[]`
- [X] `Caption[]`
- [X] `Classify[]`
- [ ] `Compare[]`
- [ ] `ComparePose[]`
- [X] `Count[]`
- [ ] `CountInRegion[]`
- [ ] `Cutout[]`
- [X] `Describe[]`
- [X] `Detect[]`
- [ ] `DetectPose[]`
- [ ] `Exit[]`
- [ ] `FilterByClass[]`
- [ ] `Find[]`
- [X] `First[]`
- [ ] `GetDistinctScenes[]`
- [ ] `GetEdges[]`
- [ ] `GetFPS[]`
- [ ] `GetText[]`
- [X] `Greyscale[]`
- [ ] `If[]`
- [X] `Import[]`
- [X] `In[]` (folder of images)
- [ ] `In[]` (video file)
- [ ] `Input[]`
- [X] `Last[]`
- [X] `Load[]`
- [ ] `Make[]`
- [X] `Not[]`
- [ ] `Paste[]`
- [ ] `PasteRandom[]`
- [ ] `Profile[]`
- [X] `Random[]`
- [ ] `Read[]`
- [ ] `ReadQR[]`
- [ ] `Replace[]`
- [X] `Reset[]`
- [ ] `Resize[]`
- [ ] `Rotate[]`
- [ ] `Save[]`
- [X] `Say[]`
- [ ] `Search[]`
- [ ] `Segment[]`
- [ ] `Select[]`
- [ ] `SetBrightness[]`
- [ ] `SetConfidence[]`
- [ ] `Show[]`
- [ ] `ShowText[]`
- [X] `Similarity[]`
- [ ] `Size[]`
- [X] `Use[]`
- [ ] `UseCamera[]`
- [ ] `Web[]`