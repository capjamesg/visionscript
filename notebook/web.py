import random

import lang
from flask import Flask, jsonify, render_template, request
from lang import parser

app = Flask(__name__)

notebooks = {}


def init_notebook():
    # cells have a session that contains state and an output
    return {"session": None, "cells": []}


@app.route("/notebook", methods=["GET", "POST"])
def notebook():
    # generate random id, then redirect user
    if request.method == "POST":
        data = request.json
        user_input = data["code"]

        session_id = data["state_id"]

        if session_id not in notebooks:
            session = lang.VisionScript()

            notebooks[session_id]["session"] = session

        session = notebooks[session_id]["session"]

        try:
            session.parse(parser.parse(user_input.strip() + "\n"))
        except Exception as e:
            raise e
            return jsonify({"error": str(e)})

        notebooks[int(state_id)]["cells"].append(user_input)

        return jsonify({"output": session.output})

    state_id = random.randint(1, 100000)
    notebooks[state_id] = init_notebook()

    return render_template("notebook.html", state_id=state_id)


if __name__ == "__main__":
    app.run(debug=True)
