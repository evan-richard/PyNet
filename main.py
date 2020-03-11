#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    main.py

    Flask RESTful api to handle requests to
	and from the Brain
'''

from flask import Flask, jsonify, request
from flask_cors import CORS
from Brain import Brain
from ThemeHelper import build_theme_id


app = Flask(__name__)
CORS(app)


@app.route('/mainpage/')
def mainpage():
    return jsonify('TEST')


@app.route('/themes', methods=['GET'])
def get_themes():

    return jsonify(container[int(number)])


@app.route('/train-theme', methods=['POST'])
def train_theme():
    body = request.json
    theme_id_lst = build_theme_id(body.get('theme'))
    print(theme_id_lst)
    return jsonify(request.json)


@app.route('/delete/<number>')
def delete(number):
    if int(number) > len(container):
        return 'That number is greater than the size of the container'
    else:
        container.pop(int(number))
        return 'Success'


if __name__ == '__main__':
    app.run(debug=True)
