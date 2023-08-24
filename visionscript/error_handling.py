from spellchecker import SpellChecker

from visionscript.usage import (language_grammar_reference,
                                lowercase_language_grammar_reference)

spell = SpellChecker()


def handle_unexpected_characters(e, code, interactive=False):
    # if line doesn't end with ], add it
    if not code.strip().endswith("]"):
        code += "]"

        return

    # if space between statement and [, remove it
    # get position of [
    position = code.find("[")

    if code[position - 1] == " ":
        code = code[: position - 1] + code[position:]

        return

    # replace all “ with "
    code = code.replace("“", '"')
    code = code.replace("”", '"')

    # raise error if character not in grammar
    if e.char not in ["[", "]", "'", '"', ",", " ", '"', '"', "\n", "\t", "\r"]:
        print(f"Syntax error on line {e.line}, column {e.column}.")
        print(f"Unexpected character: {e.char!r}")
        exit(1)

    # raise error if class doesn't exist
    line = e.line
    column = e.column

    # check if function name in grammar
    function_name = code.strip().split("\n")[line - 1].split("[")[0].strip()

    language_grammar_reference_keys = language_grammar_reference.keys()

    if function_name in language_grammar_reference_keys:
        print(f"Syntax error on line {line}, column {column}.")
        print(f"Unexpected character: {e.char!r}")
        exit(1)

    spell.known(lowercase_language_grammar_reference)
    spell.word_frequency.load_words(lowercase_language_grammar_reference)

    alternatives = spell.candidates(function_name)

    if len(alternatives) == 0:
        print(f"Function {function_name} does not exist.")
        exit(1)

    print(f"Function '{function_name}' does not exist. Did you mean one of these?")
    print("-" * 10)

    for item in list(alternatives):
        if item.lower() in lowercase_language_grammar_reference:
            print(
                list(language_grammar_reference.keys())[
                    lowercase_language_grammar_reference.index(item.lower())
                ]
            )

    if interactive is False:
        exit(1)


def handle_unexpected_token(e, interactive=False):
    line = e.line
    column = e.column

    print(f"Syntax error on line {line}, column {column}.")
    print(f"Unexpected token: {e.token!r}")
    if interactive is False:
        exit(1)
