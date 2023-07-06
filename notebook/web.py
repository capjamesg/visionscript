import random

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

import lang
from lang import init_state, parse_tree, parser

notebooks = {}


def init_notebook():
    # cells have a state and an output
    return {"state": init_state(), "cells": []}


@app.route("/notebook", methods=["GET", "POST"])
def notebook():
    # generate random id, then redirect user
    if request.method == "POST":
        # data is jjson
        data = request.json
        user_input = data["code"]

        state_id = data["state_id"]

        state = notebooks[int(state_id)]["state"]

        lang.state = state

        try:
            print(user_input.strip())
            parse_tree(parser.parse(user_input.strip() + "\n"), state)
            print(lang.state)
        except Exception as e:
            raise e
            return jsonify({"error": str(e)})

        notebooks[int(state_id)]["cells"].append(user_input)

        notebooks[int(state_id)]["state"] = lang.state

        return jsonify({"output": state["output"]})

    state_id = random.randint(1, 100000)
    notebooks[state_id] = init_notebook()

    return render_template("notebook.html", state_id=state_id)


if __name__ == "__main__":
    app.run(debug=True)
