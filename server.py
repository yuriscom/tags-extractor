import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from text_keyword_extractor import TagsExtractor
from htmlparser import MLStripper

# import subprocess

app = Flask(__name__)
CORS(app)


@app.route('/api/tags', methods=['POST'])
def get_keywords():
    text = request.json.get("text")
    s = MLStripper()
    s.feed(text)
    strippedText = s.get_data()

    tex = TagsExtractor()
    tags = tex.extract(strippedText, num=20)

    return jsonify(tags=tags)
