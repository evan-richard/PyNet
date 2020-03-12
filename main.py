#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    main.py

    Flask RESTful api to handle requests to
    and from the Brain
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from Brain import Brain
from ThemeHelper import build_theme_id, run_sample_themes, rgb_to_theme


app = Flask(__name__)
CORS(app)


@app.route("/train", methods=["POST"])
def train_theme_data():
    body = request.json
    brain = Brain()
    data_set_lst = []
    # Build training data
    theme_list = list(body.get("data", []))
    print(theme_list)
    for theme_data in theme_list:
        theme_id_lst = build_theme_id(theme_data.get("theme"))
        data_set = {
            "input": theme_id_lst,
            "output": [round(int(theme_data.get("score")) / 4, 4)],
        }
        data_set_lst.append(data_set)

    # Train the aggregated theme data
    brain.train(data_set_lst, iterations=10000)
    print("Done")

    # Recommend a list of sorted themes based on trained data
    theme_results = run_sample_themes(brain)
    theme_results = sorted(theme_results, key=lambda i: i["score"], reverse=True)
    results = rgb_to_theme(theme_results[:20])

    # Return the 20 most highly recommended themes
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
