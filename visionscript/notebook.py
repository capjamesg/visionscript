import copy
import json
import os
import string
import time
import uuid
import numpy as np

import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for

import visionscript.lang as lang
from visionscript.lang import parser

app = Flask(__name__)

API_URL = None

notebooks = {}


def init_notebook():
    # cells have a session that contains state and an output
    # notebook schema looks like:
    # {
    #   "session": session,
    #   "cells": [
    #       {
    #           "type": "code",
    #           "data": "code"
    #       },
    #       {
    #           "type": "comment",
    #           "data": "comment"
    #       }
    #   ],
    #   "output": [
    #       "output",
    #       ...
    #   ]
    # }
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
        is_text_cell = data.get("is_text_cell", False)

        user_input = data["code"]

        if (
            notebooks.get(session_id) is None
            or notebooks[session_id].get("session") is None
        ):
            session = lang.VisionScript()

            session.notebook = True

            notebooks[session_id]["session"] = session

        session = notebooks[session_id]["session"]

        if is_text_cell:
            notebooks[session_id]["cells"].append(
                {"type": "editable_text", "data": user_input}
            )
            notebooks[session_id]["output"].append(
                {"type": "editable_text", "data": ""}
            )

            # save notebook
            with open(os.path.join("tmp", session_id + ".vicnb"), "w") as f:
                json.dump(
                    {
                        "cells": notebooks[session_id]["cells"],
                        "output": notebooks[session_id]["output"],
                    },
                    f,
                )

            return jsonify({"output": "", "time": 0})

        start_time = time.time()

        code = parser.parse(user_input.strip() + "\n")

        session.check_inputs(code)

        if len(session.state["input_variables"]) == 0:
            try:
                session.parse_tree(code)
            except Exception as e:
                raise e
                return jsonify({"error": str(e)})

        end_time = time.time()

        run_time = round(end_time - start_time, 1)

        notebooks[session_id]["cells"].append({"type": "code", "data": user_input})
        notebooks[session_id]["output"].append(session.state["output"])

        # if output is ndarray, convert to base64 image
        if isinstance(session.state["output"].get("text"), np.ndarray):
            import base64
            from io import BytesIO

            image = BytesIO()
            # load from np array
            from PIL import Image

            Image.fromarray(session.state["output"]["text"]).save(image, format="PNG")

            notebooks[session_id]["output"][-1] = {
                "image": base64.b64encode(
                image.getvalue()
            ).decode("utf-8"),
                "type": "image",
            }

        # save notebook
        with open(os.path.join("tmp", session_id + ".vicnb"), "w") as f:
            json.dump(
                {
                    "cells": notebooks[session_id]["cells"],
                    "output": notebooks[session_id]["output"],
                },
                f,
            )

        return jsonify({"output": notebooks[session_id]["output"][-1], "time": run_time})

    if request.args.get("state_id"):
        state_id = request.args.get("state_id")
    else:
        state_id = uuid.uuid4().hex

    notebooks[state_id] = init_notebook()

    return render_template(
        "notebook.html", state_id=state_id, api_url=API_URL or request.url_root
    )


@app.route("/notebook/upload", methods=["POST"])
def upload():
    from werkzeug.utils import secure_filename

    session_id = request.args.get("state_id")
    file = request.files["file"]

    file.filename = secure_filename(file.filename)

    if session_id and notebooks.get(session_id) is None:
        return jsonify({"error": "No session found"}), 404

    # if file is taken
    if os.path.exists(os.path.join("tmp", file.filename)):
        # add unique id
        while os.path.exists(os.path.join("tmp", file.filename)):
            file.filename = uuid.uuid4().hex[:4] + file.filename

    # save as tmp file
    file_name = file.filename

    # only allow image uploads
    import mimetypes

    if file_name == "":
        return jsonify({"error": "No file provided"})

    if mimetypes.guess_type(file_name)[0]:
        if not mimetypes.guess_type(file_name)[0].startswith(
            "text"
        ) and not mimetypes.guess_type(file_name)[0].startswith("image"):
            return jsonify({"error": "File type not allowed"}), 415
    elif not file_name.endswith(".vicnb"):
        return jsonify({"error": "File type not allowed"}), 415

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
    session_id = request.json.get("state_id")
    file_name = "export.vic"

    if session_id and notebooks.get(session_id) is None:
        return jsonify({"error": "No session found"})

    if file_name is None:
        return jsonify({"error": "No file name provided"})

    notebook = copy.deepcopy(notebooks[session_id])

    # delete session
    del notebook["session"]

    return jsonify({"file": notebook})


@app.route("/notebook/deploy", methods=["POST"])
def deploy():
    session_id = request.json.get("state_id")
    name = request.json.get("name")
    api_url = request.json.get("api_url")
    api_key = request.json.get("api_key")
    description = request.json.get("description")
    publish_as_noninteractive_webpage = request.json.get(
        "publish_as_noninteractive_webpage"
    )

    if session_id and notebooks.get(session_id) is None:
        return jsonify({"error": "No session found"}), 404

    if name is None:
        return jsonify({"error": "No file name provided"}), 400

    # make a post request
    notebook = copy.deepcopy(notebooks[session_id])

    app_slug = name.translate(
        str.maketrans("", "", string.punctuation.replace("-", ""))
    ).replace(" ", "-")

    deploy_request = requests.post(
        api_url,
        json={
            "title": name,
            "slug": app_slug,
            "api_key": api_key,
            "description": description,
            "script": "\n".join([cell["data"] for cell in notebook["cells"]]),
            "notebook": notebook["cells"],
            "output": notebook["output"],
            "variables": notebook["session"].state["input_variables"],
            "publish_as_noninteractive_webpage": publish_as_noninteractive_webpage,
        },
    )

    if deploy_request.ok:
        return jsonify({"success": True, "message": deploy_request.json()["id"]})

    return jsonify({"success": False, "message": deploy_request.text})


@app.route("/static/<path:path>")
def static_files(path):
    return app.send_static_file(path)


@app.route("/quit")
def quit():
    exit()
