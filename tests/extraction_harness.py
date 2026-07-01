"""Shared extraction harness for the characterization tests.

This mirrors the live Lambda flow (see ``lmbda.lambda_handler``): sanitize the raw
text, strip HTML, then run ``TagsExtractor.extract`` with the default label weights.
It is the single source of truth for both golden generation and the test, so the two
can never drift apart.
"""
import os
import sys
from pathlib import Path

# Environment must be configured before importing the spaCy loader, which reads
# these variables at import time.
os.environ.setdefault("ENV", "lcl")
os.environ.setdefault("MODEL", "en_core_web_md")
os.environ.setdefault(
    "SOURCE_LINK", "https://github.com/explosion/spacy-models/releases/download"
)

# Make the project modules importable regardless of where the caller runs from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from spacy_wrapper import SpacyWrapper  # noqa: E402
from text_keyword_extractor import TagsExtractor  # noqa: E402
from htmlparser import MLStripper  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Fixed inputs and parameters used to pin behavior. These match the defaults in
# local_lambda.py so the golden output reflects the real pipeline.
INPUT_FILES = ["content.txt", "content2.txt", "content3.txt"]
LABELS = {
    "PERSON": 5,
    "ORG": 5,
    "NORP": 3,
    "GPE": 1.5,
    "EVENT": 5,
    "PRODUCT": 5,
    "WORK_OF_ART": 5,
}
NUM = 20


def _sanitize(text):
    return text.replace("\\n", " ").replace("\\r", " ")


def _strip_html(text):
    stripper = MLStripper()
    stripper.feed(text)
    return stripper.get_data()


def load_nlp():
    """Load the spaCy model once. Reuse across all extractions."""
    return SpacyWrapper().init()


def extract_for_file(nlp, filename):
    """Run the full live pipeline for one input file and return an ordered dict."""
    text = (DATA_DIR / filename).read_text(encoding="utf-8")
    stripped = _strip_html(_sanitize(text))
    extractor = TagsExtractor(nlp)
    tags = extractor.extract(stripped, LABELS, NUM)
    return dict(tags)


def golden_path(filename):
    return GOLDEN_DIR / f"{filename}.json"
