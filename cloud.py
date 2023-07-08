from visionscript import lang, parser
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, redirect
from io import BytesIO
import json
import uuid

app = Flask(__name__)

API_KEY = uuid.uuid4().hex

script = """
Load[Input["img"] ]
GetText[]
Say[]
"""

with open("scripts.json", "r") as f:
    scripts = json.load(f)

for script in scripts:
    scripts[script]["session"] = lang.VisionScript()

@app.route("/")
def index_page():
    return redirect("https://visionscript.dev")

@app.route("/<id>", methods=["GET", "POST"])
def home(id):
    if request.method == "POST":
        data = request.form
        files = request.files

        print(data, files)

        results = {}

        for variable in scripts[id]["variables"]:
            if not data.get(variable) and not files.get(variable):
                return jsonify({"error": f"Missing variable {variable}"})
            # if data is an image, turn into numpy array
            elif scripts[id]["variables"][variable] == "image":
                # buffer to numpy
                import cv2
                from PIL import Image
                
                ram_file = BytesIO()

                files[variable].save(ram_file)

                ram_file.seek(0)

                image = Image.open(ram_file)
                
                results[variable] = np.array(image)
            else:
                results[variable] = data[variable]

        try:
            session = scripts[id]["session"]
            
            session.state = {**session.state, **results}
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

    image_inputs = [[v, k] for k, v in scripts[id]["variables"].items() if v == "image"]
    text_inputs = [[v, k] for k, v in scripts[id]["variables"].items() if v == "text"]

    return render_template("index.html", id=id, image_inputs=image_inputs, text_inputs=text_inputs, title=scripts[id]["title"])

@app.route("/create", methods=["POST"])
def create():
    data = request.json
    
    if data.get("api_key") != API_KEY:
        return jsonify({"error": "Invalid API key"})
    
    id = uuid.uuid4().hex

    scripts[id] = {
        "title": data["title"],
        "script": data["script"],
        "variables": data["variables"]
    }

    scripts[id]["session"] = lang.VisionScript()

    with open("scripts.json", "w") as f:
        json.dump(scripts, f, indent=4)

    return jsonify({"id": id})

if __name__ == "__main__":
    app.run(debug=True)