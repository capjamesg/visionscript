from lang import parse_tree
import os

TEST_DIR = "./tests/"

for file in os.listdir(TEST_DIR):
    if file.endswith(".vic"):
        print(f"Testing {file}...")
        
        with open(TEST_DIR + file, "r") as f:
            parse_tree(f.read())

        print(f"Test {file} passed!")