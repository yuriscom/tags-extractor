import datetime
import json

from htmlparser import MLStripper
from spacy_wrapper import SpacyWrapper
from text_keyword_extractor import TagsExtractor, to_enrichment_dict, to_enrichment_columns

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
    extract_type = json_body.get("extract_type") or "tags"
    # labels: https://spacy.io/api/annotation#dependency-parsing
    # label dictionaries: https://github.com/explosion/spacy-models/tree/master/meta
    # use spacy.explain("NORP") to get description of each label
    labels = json_body.get("labels") or []
    num = json_body.get("num") or 20
    s = MLStripper()
    s.feed(text)
    stripped_text = s.get_data()

    tex = TagsExtractor(nlp)

    if extract_type == "tags":
        tex.extract(stripped_text, labels, num)
        result = tex.to_json()
    else:
        data = tex.extract_features(stripped_text, labels, num)
        result = json.dumps(to_enrichment_columns(data), ensure_ascii=False)

    difftime = datetime.datetime.now() - starttime
    seconds = difftime.total_seconds()
    print("tags extraction time in seconds:", seconds)

    return result



def sanitize(txt):
    return txt.replace("\\n", " ").replace("\\r", " ")
