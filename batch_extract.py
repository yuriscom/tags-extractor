#!/usr/bin/python
"""
Batch tags extraction from a CSV file.
Reads title + content columns, runs extraction on each article,
and writes results to an output CSV.

Usage:
    python batch_extract.py --input path/to/input.csv --output path/to/output.csv
"""
import os

os.environ["ENV"] = "lcl"
os.environ["MODEL"] = "en_core_web_md"

import argparse
import csv
import json
import sys
import datetime

from spacy_wrapper import SpacyWrapper
from text_keyword_extractor import TagsExtractor
from htmlparser import MLStripper

LABELS = {"PERSON": 5, "ORG": 5, "NORP": 3, "GPE": 1.5, "EVENT": 5, "PRODUCT": 5, "WORK_OF_ART": 5}
NUM_TAGS = 20


def strip_html(text):
    s = MLStripper()
    s.feed(text)
    return s.get_data()


def sanitize(text):
    return text.replace("\\n", " ").replace("\\r", " ")


def extract_tags(nlp, text, num=NUM_TAGS):
    text = sanitize(strip_html(text))
    tex = TagsExtractor(nlp)
    tags = tex.extract(text, LABELS, num)
    return tags


def main():
    parser = argparse.ArgumentParser(description="Batch tags extraction from CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--title-col", default="title", help="Name of the title column (default: title)")
    parser.add_argument("--content-col", default="content", help="Name of the content column (default: content)")
    parser.add_argument("--id-col", default="original_article_id", help="Name of the ID column (default: article_id)")
    parser.add_argument("--num", type=int, default=NUM_TAGS, help="Max number of tags per article (default: 20)")
    args = parser.parse_args()

    print("Loading NLP model...")
    nlp = SpacyWrapper().init()
    print("Model loaded.\n")

    with open(args.input, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    total = len(rows)
    print(f"Processing {total} articles...\n")

    results = []
    for i, row in enumerate(rows, 1):
        article_id = row.get(args.id_col, "")
        title = row.get(args.title_col, "") or ""
        content = row.get(args.content_col, "") or ""
        text = f"{title}. {content}".strip()

        start = datetime.datetime.now()
        try:
            tags = extract_tags(nlp, text, args.num)
            top_tags = ", ".join(list(tags.keys())[:10])
            tags_json = json.dumps(tags)
            status = "ok"
        except Exception as e:
            tags_json = "{}"
            top_tags = ""
            status = f"error: {e}"

        elapsed = (datetime.datetime.now() - start).total_seconds()
        print(f"[{i}/{total}] {title[:60]!r:64} → {top_tags[:60]}  ({elapsed:.2f}s)  [{status}]")

        results.append({
            args.id_col: article_id,
            "text": text,
            "tags_json": tags_json,
            "top_tags": top_tags,
            "status": status,
        })

    with open(args.output, "w", newline="", encoding="utf-8") as fout:
        fieldnames = [args.id_col, "text", "tags_json", "top_tags", "status"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
