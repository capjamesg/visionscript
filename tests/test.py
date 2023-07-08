import visionscript as lang
import os
import tqdm

TEST_DIR = "./"

tests = os.listdir(TEST_DIR)

passed = 0
test_count = len(tests)

for file in tqdm.tqdm(os.listdir(TEST_DIR)): 
    session = lang.VisionScript()

    if file.endswith(".vic"):
        print(f"Testing {file}...")
        
        try:
            with open(TEST_DIR + file, "r") as f:
                session.parse_tree(lang.parser.parse(f.read() + "\n"))
        except Exception as e:
            print(f"Test {file} failed!")
            print(e)
            continue

    passed += 1

print(f"{passed}/{test_count} tests passed!")