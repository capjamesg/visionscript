# Syntax correction for text run through paper OCR

import pygtrie
import os
from visionscript.grammar import grammar

CONTEXT_MANAGERS = ("If[", "In[", "UseCamera[")
INDENDATION_MANAGER = "Next["

lines = grammar.split("\n")

functions = [l.split(" ")[1].replace('"', "") for l in lines if len(l.split(" ")) > 1 and l.split(" ")[1].istitle()]

functions = ["".join([c for c in f if c.isalnum() or c == "." or c == "_"]) for f in functions]

# if < 2 letters long, remove (these are false positives)
functions = [f for f in functions if len(f) >= 2]

# get longest prefix, then chomp
trie = pygtrie.CharTrie()

for f in functions:
    trie[f] = True

def line_processing(string: str) -> str:
    context_level = 0

    lines = string.split("\n")

    all_lines = []

    for l in lines:
        if context_level > 0:
            l = "\t" * context_level + l

        # if Next, -1 context
        if l.strip().startswith(INDENDATION_MANAGER):
            context_level -= 1
            if context_level < 0:
                context_level = 0

        if l.startswith(CONTEXT_MANAGERS):
            context_level += 1

        all_lines.append(l)

    return "\n".join(all_lines)

def syntax_correction(string: str) -> str:
    string = "Load[Detect[dog]Replace[Calpng]"
    string = "Load['test.png']\nDetect['dog']\nReplace['cat']\n"

    final_string = ""

    while True:
        longest_prefix, _ = trie.longest_prefix(string)
        # if longest prefix is none, strip chars until we find a prefix
        if len(string) == 0 and longest_prefix == None:
            break

        if longest_prefix == None:
            string = string[1:]
            continue

        if longest_prefix == "":
            break
        else:
            final_string += longest_prefix
            # if ] before next function, then we are done
            next_longest_prefix, _ = trie.longest_prefix(string[len(longest_prefix) :])

            # if len between longest_prefix and next_longest_prefix is 0, then add a ] and move on
            if next_longest_prefix == None:
                final_string += string[len(longest_prefix) :]
                # if not ends with ], then add one
                if not final_string.endswith("]"):
                    final_string += "]"
                break

            longest_prefix_idx = string.find(longest_prefix)
            next_longest_prefix_idx = string.find(next_longest_prefix)

            #print("Longest prefix idx: " + str(longest_prefix_idx), next_longest_prefix_idx)

            if (longest_prefix_idx + len(longest_prefix)) == next_longest_prefix_idx:
                final_string += "]\n"
                string = string[len(longest_prefix) :]
                continue

            text_between_current_and_next = string[len(longest_prefix) :].split("[")[1]
            #print("Text between current and next: " + text_between_current_and_next)

            text_between_current_and_next = text_between_current_and_next.split("]")[0]

            #print("Text between current and next: " + text_between_current_and_next)

            final_string += text_between_current_and_next + "]\n"
            string = string[len(longest_prefix) + len(text_between_current_and_next) + 2 :]

    final_string = final_string.replace("]", "]\n")

    # chomp all new lines from end
    while final_string.endswith("\n") or final_string.endswith("]"):
        final_string = final_string[:-1]

    final_string += "]"

    return final_string