# tags-extractor

Keyword/tag extraction service built on spaCy's named entity recognition and a TextRank (PageRank-based) weighting algorithm. Designed to run as an AWS Lambda function, with a local simulation mode for development.

* Python 3.11+

---

## Local run

#### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download the spaCy language model
```bash
python -m spacy download en_core_web_md
```
This installs the model version that matches your installed spaCy automatically.

#### 4. Run
```bash
python local_lambda.py
```

The script reads from `data/content2.txt` — replace its contents with whatever article text you want to extract tags from.

---

## AWS Lambda deployment

#### Lambda packaging files
* `htmlparser.py`
* `lmbda.py`
* `spacy_module_loader.py`
* `spacy_wrapper.py`
* `text_keyword_extractor.py`

#### Deploy commands
```bash
zip tags-extractor.zip htmlparser.py lmbda.py spacy_module_loader.py spacy_wrapper.py text_keyword_extractor.py

aws --profile=<your-profile> s3 cp tags-extractor.zip s3://<your-bucket>

aws lambda update-function-code \
  --profile=<your-profile> \
  --region us-east-1 \
  --function-name <your-function-arn> \
  --s3-bucket <your-bucket> \
  --s3-key tags-extractor.zip \
  --publish
```

#### Lambda environment variables
| Variable | Example value | Notes |
|---|---|---|
| `MODEL` | `en_core_web_md-3.7.1` | Versioned model name |
| `MODEL_PATH` | `/mnt/models` | EFS mount path where the model is stored |
| `SOURCE_LINK` | `https://github.com/explosion/spacy-models/releases/download` | Used to auto-download the model if not found at MODEL_PATH |
| `SOURCE_BUCKET` | *(unused)* | Legacy, no longer used |
| `ENV` | `dev` / `prd` | Environment flag |

#### Lambda layer (spaCy)

spaCy depends on NumPy and must be compiled for Linux, so the layer needs to be built inside a Linux container.

```bash
# Build and enter a Linux container
cd layer/spacy
docker build -t tags-extractor/lnx .
docker run -d -ti -v "$(pwd)":/home/docker --name lnx tags-extractor/lnx
docker exec -it lnx bash

# Inside the container — install spaCy for the target Python version
cd /home/docker
pyenv install 3.11
pyenv global 3.11
pip3 install spacy -t .
cd ..
zip -r spacy_layer.zip .
exit

# Upload and register the layer
aws --profile=<your-profile> s3 cp layer/spacy/spacy_layer.zip s3://<your-bucket>
```

Then create a new Lambda layer from the uploaded zip in the AWS console or via CLI.
