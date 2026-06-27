import os
import spacy
import urllib.request
from pathlib import Path
import tarfile

model = os.environ.get('MODEL') or "en_core_web_sm"
source_bucket = os.environ.get('SOURCE_BUCKET') or "swt-lambda-uploads"
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


def download_dir(prefix, local, bucket, client=None):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client object
    """
    if client is None:
        import boto3
        client = boto3.client('s3')
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': bucket,
        'Prefix': prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(bucket, k, dest_pathname)
