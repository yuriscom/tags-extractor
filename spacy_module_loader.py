import os
import spacy
import urllib.request
from pathlib import Path
import tarfile

model = os.environ.get('MODEL') or "en_core_web_sm"
source_link = os.environ.get('SOURCE_LINK')
data_path = os.environ.get("MODEL_PATH") or ""
env = os.environ.get("ENV") or ""


def init_nlp():
    # Local dev: no MODEL_PATH set → load from spaCy's installed package
    if data_path == "":
        model_name = model.split('-')[0]  # strip version suffix, e.g. en_core_web_md
        print(f"Loading model from installed spaCy package: {model_name}")
        return spacy.load(model_name)

    save_path = Path(data_path) / model
    if not os.path.exists(save_path):
        download(model, data_path)

    dirname = model.split('-')[0]
    nlp = spacy.load(save_path / dirname / model)
    return nlp


def download(model, data_path):
    url = f'{source_link}/{model}/{model}.tar.gz'
    print(f'Downloading... {url}')
    filename = Path(data_path) / f'{model}.tar.gz'

    res = urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename) as f:
        f.extractall(path=data_path)
