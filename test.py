import lang
import os

TEST_DIR = "./tests/"

for file in os.listdir(TEST_DIR):
    lang.state = lang.init_state()
    if file.endswith(".vic"):
        print(f"Testing {file}...")
        
        try:
            with open(TEST_DIR + file, "r") as f:
                lang.parse_tree(lang.parser.parse(f.read().strip()))
        except Exception as e:
            print(f"Test {file} failed!")
            print(e)
            continue

        print(f"Test {file} passed!")