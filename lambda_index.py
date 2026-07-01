import json
import datetime

from spacy_wrapper import SpacyWrapper
from text_keyword_extractor import TagsExtractor
from htmlparser import MLStripper

nlp = SpacyWrapper().init()


def lambda_handler(event, context):
    starttime = datetime.datetime.now()
    body = event.get("body") or {}

    print(body)

    if isinstance(body, dict):
        json_body = body
    else:
        json_body = json.loads(body)

    text = json_body.get("data") or ""
    text = sanitize(text)
    # labels: https://spacy.io/api/annotation#dependency-parsing
    # label dictionaries: https://github.com/explosion/spacy-models/tree/master/meta
    # use spacy.explain("NORP") to get description of each label
    labels = json_body.get("labels") or []
    num = json_body.get("num") or 20
    s = MLStripper()
    s.feed(text)
    stripped_text = s.get_data()

    tex = TagsExtractor(nlp)
    tex.extract(stripped_text, labels, num)

    tags = tex.to_json()
    difftime = datetime.datetime.now() - starttime
    seconds = difftime.total_seconds()
    print("tags extraction time in seconds:", seconds)

    return {
        'statusCode': 200,
        'body': tags
    }


def sanitize(txt):
    return txt.replace("\\n", " ").replace("\\r", " ")
