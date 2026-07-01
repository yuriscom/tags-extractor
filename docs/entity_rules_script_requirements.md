# Entity Rules Evaluation Script - Requirements and Implementation Guide

## 1. Purpose

Create a small Python POC script that evaluates whether articles from the Allison data lake are relevant to a specific watched entity, starting with `Target Corporation` as the first test case.

The script should answer Question A only:

```text
Is this article about the intended entity?
```

It should not solve topic slicing, semantic search, vector ranking, Elasticsearch indexing, or downstream NLP processing yet.

The script will compare simple article-selection approaches and produce an evidence-focused output file that can be manually reviewed and used to calculate precision and recall.

## 2. Background

The current data lake contains a broad daily corpus of Unicepta/Moreover articles. Production client dashboards currently rely on upstream Metabase/Lucene queries to select articles before they enter the existing NLP pipeline.

For the lake-based flow, we need an equivalent article-selection layer over the full corpus. The first POC should test whether rule-based entity resolution can reduce false positives compared with naive text search, especially for ambiguous entities such as `Target`, `Amazon`, `Ford`, and similar common-word brand names.

The first target entity is `Target Corporation`, because naive text search for `target` produces many false positives such as:

```text
price target
target audience
transfer target
military target
target date
```

## 3. Scope

### In scope

The script should:

1. Read an article CSV file.
2. Read a YAML rules file for one entity.
3. Optionally read a CSV file containing extracted tags from the existing spaCy/TextRank tag extractor.
4. Evaluate each article against the rules.
5. Produce a scored prediction per article.
6. Include matched evidence in the output.
7. Support manual review by including enough fields to understand why the script predicted yes/no/maybe.
8. Calculate basic summary counts if a manual label column is present.

### Out of scope

The script should not:

1. Use Elasticsearch.
2. Use LanceDB.
3. Use LLMs or Bedrock.
4. Generate embeddings.
5. Process the full production lake at scale.
6. Write to S3.
7. Integrate with `data_processing_unified`.
8. Build a generic rule engine framework for all future use cases.

This is a local/offline POC script.

## 4. Expected Inputs

### 4.1 Articles CSV

The script should accept an input CSV with at least:

```text
article_id
title
content
```

However, it should also support the existing POC shape where title and content may already be concatenated into a single field:

```text
article_id
text
```

Implementation should allow column names to be configurable through CLI arguments.

Recommended CLI options:

```bash
--articles path/to/articles.csv
--id-col article_id
--title-col title
--content-col content
--text-col text
```

If `--text-col` is provided and exists, use it as the full article text.

If `--text-col` is not provided, concatenate:

```text
title + ". " + content
```

### 4.2 Rules YAML

The rules file defines the entity profile and scoring behavior.

Example:

```yaml
entity_id: target_corporation
canonical_name: Target Corporation

aliases:
  strong:
    - Target Corporation
    - Target Corp
    - Target.com
    - Target Circle
    - Shipt
    - Roundel
  weak:
    - Target

positive_context:
  - retailer
  - retail
  - store
  - stores
  - shoppers
  - Brian Cornell
  - Michael Fiddelke
  - Minneapolis-based
  - same-day delivery
  - pickup
  - fulfillment
  - supply chain

negative_phrases:
  - price target
  - target price
  - target audience
  - target date
  - transfer target
  - military target
  - target range
  - target rate
  - target level
  - target market

scoring:
  strong_alias: 3
  weak_alias: 1
  positive_context: 1
  negative_phrase: -3
  tag_entity_match: 1
  tag_context_match: 1

thresholds:
  yes: 2
  no: 0

options:
  case_sensitive: false
  word_boundary_for_aliases: true
  reject_if_negative_phrase: false
```

Notes:

- `strong` aliases are high-confidence signals by themselves.
- `weak` aliases are ambiguous and should usually require additional context.
- `negative_phrases` reduce the score or can optionally reject the article completely.
- `positive_context` adds supporting evidence.
- The rules file should be analyst-editable and easy to review.

### 4.3 Optional tags CSV

The script should optionally accept the output from the existing tag extractor.

Expected fields:

```text
article_id
tags_json
top_tags
```

Recommended CLI option:

```bash
--tags path/to/tags_output.csv
```

If provided, the script should join tags by `article_id`.

The tags are not treated as ground truth. They are additional scoring signals.

## 5. Expected Output

The script should write an output CSV with one row per article.

Required output columns:

```text
article_id
title
score
decision
predicted_entity_id
matched_strong_aliases
matched_weak_aliases
matched_positive_context
matched_negative_phrases
matched_tag_entities
matched_tag_context
reason
```

Optional output columns:

```text
actual_entity_match
is_correct
error_type
text_preview
```

### 5.1 Decision values

Allowed decision values:

```text
yes
no
maybe
```

Decision logic:

```text
if score >= thresholds.yes:
    decision = "yes"
elif score <= thresholds.no:
    decision = "no"
else:
    decision = "maybe"
```

If `reject_if_negative_phrase` is true and one or more negative phrases match:

```text
decision = "no"
```

### 5.2 Reason field

The `reason` column should be concise but human-readable.

Example:

```text
strong alias: Target Circle; positive context: retailer, Brian Cornell; no negative phrase
```

Another example:

```text
weak alias: target; negative phrase: price target
```

## 6. Matching Requirements

### 6.1 Text normalization

Before matching:

1. Convert HTML entities where possible.
2. Remove or normalize excessive whitespace.
3. Optionally lowercase text if `case_sensitive: false`.
4. Keep original text for output preview.

### 6.2 Alias matching

Use phrase matching, not substring matching.

For aliases with letters/numbers, apply word boundaries by default.

For example, weak alias `Target` should match:

```text
Target announced...
shares of Target rose...
```

But should not match inside unrelated longer tokens.

### 6.3 Negative phrase matching

Negative phrase matching can be simple case-insensitive phrase matching.

Examples:

```text
price target
target price
transfer target
target audience
```

For phase 1, no proximity logic is required.

### 6.4 Positive context matching

Positive context matching can also be simple phrase matching.

Examples:

```text
Brian Cornell
retailer
stores
Target Circle
Shipt
```

### 6.5 Tags matching

If tags are provided:

1. Parse `tags_json` if available.
2. Use tag names, not tag weights, for phase 1.
3. Check whether strong aliases, weak aliases, or positive context terms appear as tags.
4. Add scoring contribution based on `tag_entity_match` and `tag_context_match`.

Do not treat tag score/weight as a calibrated confidence score.

## 7. Scoring Requirements

The first scoring implementation should be simple and explainable.

Suggested scoring:

```text
+3 for each matched strong alias
+1 for each matched weak alias
+1 for each matched positive context item
-3 for each matched negative phrase
+1 if tags include an alias
+1 if tags include positive context
```

The implementation should keep all matched evidence, not only the score.

The script should avoid hidden logic. Every score contribution should be explainable from output columns.

## 8. Manual Evaluation Support

If the input articles CSV contains a manual label column, for example:

```text
actual_entity_match
```

with values:

```text
yes
no
```

then the script should calculate:

```text
true positives
false positives
true negatives
false negatives
precision
recall
```

For `maybe`, the first implementation can either:

1. Exclude maybe rows from precision/recall calculations, or
2. Count maybe as predicted yes only if a CLI option is set.

Recommended default:

```text
Exclude maybe rows from precision/recall.
```

The script should print a summary to stdout and optionally write a summary JSON file.

Recommended CLI option:

```bash
--summary-output path/to/summary.json
```

## 9. CLI Requirements

Example usage:

```bash
python entity_rules_eval.py \
  --articles articles.csv \
  --rules rules/target.yaml \
  --tags tags_output.csv \
  --output target_predictions.csv \
  --id-col article_id \
  --title-col title \
  --content-col content
```

For the existing POC format where text is already concatenated:

```bash
python entity_rules_eval.py \
  --articles tags_output.csv \
  --rules rules/target.yaml \
  --tags tags_output.csv \
  --output target_predictions.csv \
  --id-col article_id \
  --text-col text
```

Required CLI args:

```text
--articles
--rules
--output
```

Optional CLI args:

```text
--tags
--id-col
--title-col
--content-col
--text-col
--label-col
--summary-output
```

## 10. Suggested File Structure

```text
entity-rules-poc/
  entity_rules_eval.py
  rules/
    target.yaml
  data/
    articles.csv
    tags_output.csv
  output/
    target_predictions.csv
    target_summary.json
```

No package structure is required for the first version.

## 11. Implementation Plan

### Step 1 - Implement config loading

- Use `yaml.safe_load` for rules.
- Validate that required sections exist:
  - `entity_id`
  - `canonical_name`
  - `aliases`
  - `scoring`
  - `thresholds`

### Step 2 - Implement article loading

- Use pandas or standard `csv` module.
- For POC simplicity, pandas is acceptable.
- Create a normalized full text column.
- Preserve title and article ID in output.

### Step 3 - Implement matching helpers

Functions:

```python
def normalize_text(text: str, case_sensitive: bool) -> str:
    ...

def find_phrase_matches(text: str, phrases: list[str], word_boundary: bool) -> list[str]:
    ...
```

Matching should return the matched configured phrase names.

### Step 4 - Implement tags join

- Load tags CSV if `--tags` is provided.
- Join by article ID.
- Parse `tags_json` if available.
- Fall back to `top_tags` if `tags_json` is missing or invalid.

### Step 5 - Implement scoring

For each article:

1. Match strong aliases.
2. Match weak aliases.
3. Match positive context.
4. Match negative phrases.
5. Match tag entities/context if tags are available.
6. Calculate score.
7. Assign decision.
8. Build reason string.

### Step 6 - Write output CSV

Output one row per article with score, decision, predicted entity, and all evidence columns.

### Step 7 - Optional metrics

If `--label-col` exists:

1. Compare `decision` with actual label.
2. Calculate confusion matrix.
3. Print metrics.
4. Write summary JSON if requested.

## 12. Acceptance Criteria

The implementation is acceptable when:

1. It runs locally against the existing 138-article CSV.
2. It accepts `rules/target.yaml` and produces `target_predictions.csv`.
3. The output contains one row per input article.
4. Each row includes score, decision, and matched evidence.
5. Obvious false positives such as `price target`, `target audience`, and `transfer target` are marked as `no` or low-score `maybe`.
6. Obvious Target Corporation articles with strong aliases or positive context are marked as `yes`.
7. Manual review is possible without opening the full article text for every row.
8. The scoring logic is easy to tune by editing the YAML file only.

## 13. Non-Functional Requirements

### Simplicity

The script should be easy to read and modify. Prefer straightforward code over abstraction.

### Explainability

Every decision must be traceable to matched evidence.

### Analyst-editability

The rules file should be readable by a non-engineer or data analyst.

### POC performance

The first version only needs to handle hundreds or thousands of articles locally. It does not need to handle millions of articles per day.

### Future scalability

Do not design production-scale architecture now, but avoid choices that obviously block future scale.

For example:

- Keep matching logic separate from CLI parsing.
- Keep rules external in YAML.
- Keep output evidence structured.

## 14. Future Enhancements

Possible later enhancements:

1. Support multiple entities in one run.
2. Add proximity rules, for example `Target` within 20 words of `retailer`.
3. Support regex rules.
4. Support negative context by category, for example finance, sports, military.
5. Use spaCy entities directly, not only extracted tags.
6. Use GLiNER for NER-assisted matching.
7. Add LLM adjudication for `maybe` rows.
8. Store resolved annotations into Parquet.
9. Index resolved annotations into Elasticsearch/OpenSearch.
10. Add vector search for Question B, after Question A is reliable.

## 15. Key Design Principle

Do not try to make the script appear smarter than it is.

For phase 1, the goal is an explainable, tunable baseline:

```text
alias match + positive evidence - negative evidence + optional tag evidence
```

This is enough to test whether deterministic entity resolution improves over naive text search for ambiguous brand names.
