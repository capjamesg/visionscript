import copy
import json
import os
import time
import uuid

from flask import Flask, jsonify, redirect, render_template, request, url_for

import visionscript.lang as lang
from visionscript.lang import parser

app = Flask(__name__)

notebooks = {}


def init_notebook():
    # cells have a session that contains state and an output
    return {"session": None, "cells": [], "output": []}


@app.route("/")
def home():
    return redirect(url_for("notebook"))


@app.route("/notebook", methods=["GET", "POST"])
def notebook():
    # generate random id, then redirect user
    if request.method == "POST":
        data = request.json
        session_id = data["state_id"]

        user_input = data["code"]

        if (
            notebooks.get(session_id) is None
            or notebooks[session_id].get("session") is None
        ):
            session = lang.VisionScript()

            session.notebook = True

            notebooks[session_id]["session"] = session

        session = notebooks[session_id]["session"]

        start_time = time.time()

        code = parser.parse(user_input.strip() + "\n")

        try:
            session.parse_tree(code)
        except Exception as e:
            raise e
            return jsonify({"error": str(e)})

        end_time = time.time()

        run_time = round(end_time - start_time, 1)

        notebooks[session_id]["cells"].append(user_input)
        notebooks[session_id]["output"].append(session.state["output"])

        return jsonify({"output": session.state["output"], "time": run_time})

    if request.args.get("state_id"):
        state_id = request.args.get("state_id")
    else:
        state_id = uuid.uuid4().hex

    notebooks[state_id] = init_notebook()

    return render_template("notebook.html", state_id=state_id)


@app.route("/notebook/upload", methods=["POST"])
def upload():
    session_id = request.args.get("state_id")
    file = request.files["file"]

    if session_id and notebooks.get(session_id) is None:
        return jsonify({"error": "No session found"})

    # save as tmp file
    file_name = file.filename

    # only allow image uploads
    import mimetypes

    if file_name == "":
        return jsonify({"error": "No file provided"})
    
    if mimetypes.guess_type(file_name)[0]:
        if not mimetypes.guess_type(file_name)[0].startswith("text") and not mimetypes.guess_type(file_name)[0].startswith("image"):
            return jsonify({"error": "File type not allowed"})
    elif not file_name.endswith(".vicnb"):
        return jsonify({"error": "File type not allowed"})
    
    # remove special chars
    file_name = "".join([c for c in file_name if c.isalnum() or c == "." or c == "_"])

    file_name = file_name.replace("..", "")

    # mkdir tmp if not exists
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    with open(os.path.join("tmp", file_name), "wb") as f:
        f.write(file.read())

    # if filename ends in .vicnb, reset state
    if file_name.endswith(".vicnb"):
        notebooks[session_id] = init_notebook()

        notebooks[session_id]["session"] = lang.VisionScript()
        notebooks[session_id]["session"].notebook = True

        with open(os.path.join("tmp", file_name), "r") as f:
            # file is json
            notebook = json.load(f)

            notebooks[session_id]["cells"] = notebook["cells"]
            notebooks[session_id]["output"] = notebook["output"]

            # zip cells and output
            result = []
            for cell, output in zip(notebook["cells"], notebook["output"]):
                result.append({"cell": cell, "output": output})

            return jsonify({"cells": result})

    return jsonify({"file_name": os.path.join("tmp", file_name)})


# save
@app.route("/notebook/save", methods=["POST"])
def save():
    session_id = request.args.get("state_id")
    file_name = "export.vic"

    if session_id and notebooks.get(session_id) is None:
        return jsonify({"error": "No session found"})

    if file_name is None:
        return jsonify({"error": "No file name provided"})

    notebook = copy.deepcopy(notebooks[session_id])

    # delete session
    del notebook["session"]

    with open(file_name, "w") as f:
        json.dump(notebook, f)

    return jsonify({"file": notebook})


@app.route("/static/<path:path>")
def static_files(path):
    return app.send_static_file(path)


@app.route("/quit")
def quit():
    exit()
