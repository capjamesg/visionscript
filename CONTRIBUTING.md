# Contribute to VisionScript

*Before you read on, please read the project README in full. The README explains some additional background on VisionScript you will need before contributing.*

Thank you for your interest in contributing to VisionScript!

The aim of VisionScript is to provide an abstract programming language for experimenting with computer vision. VisionScript provides powerful primitives that, combined, enable people to express their creativity.

VisionScript is a programming language. VisionScript Notebooks and Deploy, bundled with the core VisionScript project, are consumers of the language. The former lets you write and run code in an interactive web interface on desktop and mobile devices. The latter lets you deploy an API that runs VisionScript code.

## Contributing to VisionScript

There are many ways you can help contribute to VisionScript, including:

1. Adding a new function or primative to the programming language;
2. Improving the VisionScript Notebooks web interface;
3. Improving the VisionScript deployment option;
4. Improve the documentation (see the [visionscript-docs](https://github.com/capjamesg/visionscript) repository for the project documentation);
5. Add tests for existing functions and logic;
6. Reporting bugs, and;
7. Sharing ideas on how we can improve the VisionScript language, Notebooks, and deployment.

**Before you add a new feature, please open an Issue on the project repository so we can discuss and refine the idea before implementation.**

To get started with a development environment, first clone the VisionScript GitHub repository:

```
git clone https://github.com/capjamesg/visionscript
```

Then, install the `visionscript` package and its dependencies:

```
pip3 install -e .
```

Once you have installed VisionScript, you can start working on the language, Notebooks, or deployment:

```
visionscript --repl # create a REPL
visionscript --notebook # run an interactive notebook
visionscript --cloud # create a HTTP server to which you can deploy VisionScript code
```

## How the Language Works

VisionScript is a programming language. Suppose a user writes a program like this:

```
Load["./person.png"]
Detect["person"]
Replace["blue"]
Show[]
```

This program would replace all people in an image with a blue box.

VisionScript will first generate a syntax tree for this code. This is done using `lark`, a Python lexing library.

Here is the syntax tree associated with our program:

```
start
  expr
    load        "./person.png"
  

  expr
    detect      "person"
  

  expr
    replace     "blue"
  

  expr
    show
```

*Pro tip: You can generate a syntax tree for a program without evaluating the program by using the --debug=True parameter when executing a `.vic` file.*

This syntax tree has a `start` entry point that contains four expressions. Those expressions each contain one child (load, detect, replace, and show). If we had a more complicated program, elements of the tree may be nested. For instance, a `Load[Input["text"]]` statement will be nested like this:

```
start
  expr
    load
      input     "text"
```

Here, `input` is nested within `load`.

After VisionScript generates a syntax tree, the tree is evaluated.

The tree is evaluated in a Python function called `parse_tree`. This function will recursively traverse the tree. Statements may either update `state`, a global state that is maintained throughout program execution, or return a statement. Any statement that is explicitly returned may or may not update `state`, depending on the statement.

`parse_tree` contains a lot of logic for both fundamental program parsing as well as control flow (i.e. `If` statements, `Make` function definitions).

## Add a Function

To add a function to VisionScript, first edit `grammar.py` to add a grammar for your function. There are examples in the file already that show functions that accept zero, one, or an aribitrary number of arguments.

Then, create a new Python function in `lang.py` with the logic you want to enable. Take a look at other functions that are related to your idea within the codebase to see how functions are implemented.

Once you have implemented a function, add it to the `self.function_calls` value in the `VisionScript` `__init__` code.

After making these changes, your function will be part of the language and available when executing `.vic` files or using the VisionScript REPL.

VisionScript Notebooks is a _consumer of_ the VisionScript language, not the language itself. Thus, to add your function to the web interface, you need to make a few more changes.

Edit `visionscript/static/functions.js` and add your function as an entry under the header that makes the most sense (i.e. Input, Output, Process). Look at other entries to see how they work. Entries can accept file or text arguments.

If your function involves additional control flow, you will need to add that logic into the `visionscript/static/main.js` file.

## Code Organization

- `visionscript/lang.py`: Core language code.
- `test/test.py`: Run tests.
- `visionscript/usage.py`: Variables referenced for usage instructions in `lang.py`.
- `visionscript/grammar.py`: The VisionScript grammar.
- `visionscript/notebook.py`: The VisionScript notebook environment code.
- `visionscript/cloud.py`: The VisionScript cloud environment code.
- `tests/`: VisionScript tests.

## How to Make a Change

Whether you want to improve documentation, submit a bug fix, or add to the library, you'll first need to get our repo set up on your local machine.

First, fork the repository to your own GitHub account. This fork is where you will push your code changes.

Next, you'll need to download and configure the project on your local machine. You can do this by following the instructions in our [README](README.md) file. The README outlines how to install the library and run test cases.

Please create a new branch for your changes by using the `git checkout -b <branch-name>` command.

Once you have made a change, please run all test cases according to the instructions in the [README](README.md) file. This helps us assure the quality of the code submitted to the library.

When you are ready to submit a change, commit your changes to the branch. Then, submit a pull request to this repository.

A contributor will review your request and either approve it or provide feedback and proposed changes.

## Feedback

Do you have some feedback on how we can improve this file? Let us know by submitting an issue in the [Issues](https://github.com/capjamesg/visionscript/issues) section of this repository.