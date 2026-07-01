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

## Output: two contracts

`TagsExtractor` exposes two independent outputs. They can be used side by side.

### 1. Legacy tags map

`extract()` returns an `OrderedDict` of `{term: score}` ranked highest-first, and `to_json()`
serializes it. This is what the Lambda handler (`lambda_index.py`) and `local_lambda.py` use, so
running `local_lambda.py` still prints the flat map — that behavior is unchanged on purpose, since it
is the deployed API.

```python
extractor.extract(text, labels, num=20)   # -> {"Target Corporation": 20.0, "loyalty program": 12.4, ...}
extractor.to_json()                        # -> JSON string of the same map
```

### 2. Structured enrichment output

`extract_features()` returns an `ArticleEnrichmentResult` that keeps keyphrases and named entities as
**separate** signals, each with counts, scores, and (for entities) character offsets and sentence
context. Use this for storage/search-indexing rather than the flat map.

```python
from spacy_wrapper import SpacyWrapper
from text_keyword_extractor import TagsExtractor, to_enrichment_columns

nlp = SpacyWrapper().init()
extractor = TagsExtractor(nlp)

labels = {"PERSON": 5, "ORG": 5, "NORP": 3, "GPE": 1.5, "EVENT": 5, "PRODUCT": 5, "WORK_OF_ART": 5}
result = extractor.extract_features(text, labels, num=20)

result.keywords          # list[Keyword] — ranked keyphrases (same scoring as extract())
result.entities          # list[Entity] — spaCy entities grouped by (normalized, label)
result.enrichment_error  # None on success, else the error message (extraction never raises)

columns = to_enrichment_columns(result)   # dict of JSON-string columns, ready to store/index
```

**`keywords`** items:

```json
{ "text": "loyalty program", "normalized": "loyalty program", "type": "keyphrase",
  "score": 12.42, "count": 1, "source_fields": ["content"], "evidence": [] }
```

**`entities`** items:

```json
{ "text": "Target Corporation", "normalized": "target corporation", "label": "ORG",
  "score": 10.0, "count": 2, "label_weight": 5.0, "is_multi_word": true,
  "first_seen_field": "content", "source_fields": ["content"],
  "mentions": [ { "field": "content", "start": 240, "end": 258,
                  "text": "Target Corporation", "sentence": "Target Corporation announced ..." } ] }
```

**`to_enrichment_columns(result)`** returns these keys (each value is a JSON string except the last
three): `keywords_json`, `entities_json`, `keyword_texts_json`, `entity_texts_json`,
`entity_labels_json`, `entity_pairs_json`, `enrichment_version`, `enrichment_processed_at`,
`enrichment_error`.

For debugging, **`to_enrichment_dict(result)`** returns the whole result as one nested dict, so
`json.dumps(to_enrichment_dict(result), indent=2, ensure_ascii=False)` gives a readable, paste-ready
JSON document.

Entity label filtering/weighting, mention limits, and `num` are configurable via `ExtractorConfig`
(passed as `TagsExtractor(nlp, config=...)`).

---

## AWS Lambda deployment

#### Lambda packaging files
* `htmlparser.py`
* `lambda_index.py`
* `spacy_module_loader.py`
* `spacy_wrapper.py`
* `text_keyword_extractor.py`

#### Deploy commands
```bash
zip tags-extractor.zip htmlparser.py lambda_index.py spacy_module_loader.py spacy_wrapper.py text_keyword_extractor.py

aws --profile=<your-profile> s3 cp tags-extractor.zip s3://<your-bucket>

aws lambda update-function-code \
  --profile=<your-profile> \
  --region us-east-1 \
  --function-name <your-function-arn> \
  --s3-bucket <your-bucket> \
  --s3-key tags-extractor.zip \
  --publish
```

> **Note:** the Lambda function's **Handler** must be set to `lambda_index.lambda_handler`

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
