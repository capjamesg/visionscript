import json
import os
import string
import uuid
from io import BytesIO

import markdown
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file

from visionscript import lang, parser

app = Flask(__name__)

API_KEY = uuid.uuid4().hex

if not os.path.exists("scripts.json"):
    with open("scripts.json", "w") as f:
        json.dump({}, f)

if not os.path.exists("notebooks.json"):
    with open("notebooks.json", "w") as f:
        json.dump({}, f)

with open("scripts.json", "r") as f:
    scripts = json.load(f)

for script in scripts:
    scripts[script]["session"] = lang.VisionScript()

with open("notebooks.json", "r") as f:
    notebooks = json.load(f)

print("Your API key is", API_KEY)
print("Keep it safe and don't share it with anyone!")


@app.route("/")
def index_page():
    return render_template("deployintro.html")


@app.route("/<id>", methods=["GET", "POST"])
def home(id):
    if request.method == "POST":
        if scripts.get(id) is None:
            return jsonify({"error": "Invalid ID"})

        # if no session for the script, make it
        if scripts[id].get("session") is None:
            scripts[id]["session"] = lang.VisionScript()

        data = request.form
        files = request.files

        results = {}

        for variable in scripts[id]["variables"]:
            if not data.get(variable) and not files.get(variable):
                return jsonify({"error": f"Missing variable {variable}"})
            # if data is an image, turn into numpy array
            elif scripts[id]["variables"][variable] == "image":
                from PIL import Image

                ram_file = BytesIO()

                files[variable].save(ram_file)

                ram_file.seek(0)

                image = Image.open(ram_file).convert("RGB")

                results[variable] = np.array(image)[:, :, ::-1]
            else:
                results[variable] = data[variable]

        try:
            session = scripts[id]["session"]

            session.state["input_variables"] = {
                **session.state["input_variables"],
                **results,
            }

            session.notebook = True

            session.parse_tree(parser.parse(scripts[id]["script"]))
        except Exception as e:
            raise e
            return jsonify({"error": str(e)})

        output = session.state["output"]

        if isinstance(output, dict) and output.get("image"):
            # output is base64, convert to png
            import base64

            image = BytesIO(base64.b64decode(output["image"]))
            image.seek(0)
            return send_file(image, mimetype="image/png")

        return jsonify({"output": session.state["output"]})

    if not scripts.get(id):
        return redirect("/")

    image_inputs = [[v, k] for k, v in scripts[id]["variables"].items() if v == "image"]
    text_inputs = [[v, k] for k, v in scripts[id]["variables"].items() if v == "text"]

    return render_template(
        "index.html",
        id=id,
        image_inputs=image_inputs,
        text_inputs=text_inputs,
        title=scripts[id]["title"],
    )


@app.route("/notebook/<id>")
@app.route("/notebook/<id>/embed")
@app.route("/notebook/<id>/export_vic")
@app.route("/notebook/<id>/export_vicnb")
def notebook(id):
    with open("notebooks.json", "r") as f:
        notebooks = json.load(f)

    notebook_data = notebooks.get(id)

    if notebook_data is None:
        return redirect("/notebook")
    
    if request.path.endswith("/export_vicnb"):
        # force download with Content-Disposition so that user doesn't see raw JSON
        return jsonify(notebook_data), 200, {
            "Content-Disposition": f"attachment; filename={id}.vicnb"
        }
    elif request.path.endswith("/export_vic"):
        # concatenate all celsl
        cells = [i["data"] for i in notebook_data["notebook"]]

        code = "\n".join(cells) + "\n"

        return jsonify(code), 200, {
            "Content-Disposition": f"attachment; filename={id}.vic"
        }

    # merge cells and output
    cells = []

    for i, cell in enumerate(notebook_data["notebook"]):
        # if output has editable_text key, parse with markdown
        if cell.get("type") == "editable_text":
            cell["data"] = markdown.markdown(cell["data"])

        cells.append(
            {
                "type": "code",
                "data": cell,
                "output": notebook_data["output"][i],
                "id": i,
            }
        )

    if request.path.endswith("/embed"):
        template = "public_notebook_embed.html"
    else:
        template = "public_notebook.html"

    return render_template(
        template,
        cells=cells,
        url_root=request.url_root.strip("/"),
        title=notebook_data["title"],
        description=notebook_data["description"],
        id=id,
        notebook_url=request.url_root.strip("/") + "/notebook/" + id,
    )


@app.route("/create", methods=["POST"])
def create():
    data = request.json

    if data.get("api_key") != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401

    id = data["slug"].lower()

    publish_as_noninteractive_webpage = data.get("publish_as_noninteractive_webpage")

    if publish_as_noninteractive_webpage:
        # add to notebooks.json
        notebooks = json.load(open("notebooks.json", "r"))

        notebooks[id] = {
            "title": data["title"],
            "notebook": data["notebook"],
            "output": data["output"],
            "description": data.get("description"),
        }

        app_slug = data["title"].translate(
            str.maketrans("", "", string.punctuation.replace("-", ""))
        ).replace(" ", "-")

        notebooks[id]["app_slug"] = app_slug

        with open("notebooks.json", "w") as f:
            json.dump(notebooks, f)

        return jsonify({"id": request.url_root + "notebook/" + id})

    with open("scripts.json", "r") as f:
        scripts = json.load(f)

    scripts[id] = {
        "title": data["title"],
        "script": data["script"],
        "variables": data["variables"],
        "description": data.get("description"),
    }

    app_slug = data["title"].translate(
        str.maketrans("", "", string.punctuation.replace("-", ""))
    ).replace(" ", "-")

    scripts[id]["app_slug"] = app_slug

    with open("scripts.json", "w") as f:
        json.dump(scripts, f)

    scripts = json.load(open("scripts.json", "r"))

    return jsonify({"id": request.url_root + id})
