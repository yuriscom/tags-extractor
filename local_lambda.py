#!/usr/bin/python
import os

os.environ["ENV"] = "lcl"
os.environ["MODEL"] = "en_core_web_md"
#os.environ["MODEL_PATH"] = "/Users/yuriscom/projects/spacytest/models"
# MODEL_PATH is intentionally unset for local dev — the loader will use the installed spaCy package.
# To use a specific downloaded model instead, set MODEL_PATH to the folder containing the model dir,
# and set MODEL to the versioned name, e.g. "en_core_web_md-3.7.1".
os.environ["SOURCE_LINK"] = "https://github.com/explosion/spacy-models/releases/download"

import lambda_index


def main():
    f = open("data/content2.txt", "r")
    text = f.read()
    f.close()

    # response = lambda_index.lambda_handler({
    #     "body": {
    #         "data": text,
    #         "labels": ["PERSON", "ORG", "NORP", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART"],
    #         "num": 20
    #     }
    # }, {})

    response = lambda_index.lambda_handler({
        "body": {
            "data": text,
            "extract_type": "features", # features | tags
            # "labels": ["PERSON", "ORG", "NORP", "FAC", "EVENT", "PRODUCT", "WORK_OF_ART"],
            "labels": {"PERSON": 5, "ORG": 5, "NORP": 3, "GPE": 1.5, "EVENT": 5, "PRODUCT": 5,
                       "WORK_OF_ART": 5},
            "num": 20
        }
    }, {})

    print(response)


if __name__ == "__main__":
    main()
