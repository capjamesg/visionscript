import visionscript as lang
import os

TEST_DIR = "./tests/"

for file in os.listdir(TEST_DIR):
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

        print(f"Test {file} passed!")