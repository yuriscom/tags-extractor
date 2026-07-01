import json
import re
from collections import OrderedDict, Counter
from dataclasses import dataclass, field

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS


@dataclass
class ExtractorConfig:
    """Tunable parameters for tag extraction, passed to TagsExtractor."""
    exclude_pos: list = field(default_factory=lambda: ["PUNCT", "PART"])
    entity_label_weights: dict = field(default_factory=lambda: {
        "PERSON": 5, "ORG": 5, "NORP": 3, "GPE": 1.5, "EVENT": 5, "PRODUCT": 5,
        "WORK_OF_ART": 5, "DATE": 0,
    })
    multiple_occurrence_multiplier: bool = True
    full_term_weight_bonus: dict = field(default_factory=lambda: {"PERSON": 2})


class TextRank4Keyword():
    def __init__(self, nlp):
        self.nlp = nlp
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.doc = None

    def set_stopwords(self, stopwords):
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, labels, lower):
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with candidate POS tag
                is_candidate_pos = token.pos_ in candidate_pos and token.is_stop is False and token.is_alpha is True
                is_candidate_label = len(labels) == 0 or (len(labels) > 0 and token.ent_type_ in labels)
                if is_candidate_pos and is_candidate_label:
                    if lower is True:
                        selected_words.append(token.lemma_.lower())
                    else:
                        selected_words.append(token.lemma_)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get symmetric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, out=np.zeros_like(g), where=norm != 0)

        return g_norm

    def analyze(self, text,
                labels=[],
                candidate_pos=['NOUN', 'PROPN'],
                window_size=3, lower=False, stopwords=list()):

        # Set stop words
        self.set_stopwords(stopwords)

        # Parse text with spaCy
        doc = self.nlp(text)

        self.doc = doc

        # Filter sentences (here put the logic for what we include in the sentence and what not)
        sentences = self.sentence_segment(doc, candidate_pos, labels, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initialization for weight (pagerank value)
        pr = np.ones(len(vocab), dtype='float')

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


class TagsExtractor():
    def __init__(self, nlp, config=None):
        self.nlp = nlp
        self.config = config or ExtractorConfig()
        self.remove_ambiguity = True
        self.tags = OrderedDict()
        # Per-instance copy so extract() can merge request labels without mutating the config.
        self.entity_label_weights = self.config.entity_label_weights.copy()

    # labels: https://spacy.io/api/annotation#dependency-parsing
    def extract(self, text, labels, num=20):
        if isinstance(labels, dict):
            self.entity_label_weights.update(labels)
            labels = list(labels.keys())

        tr4w = TextRank4Keyword(self.nlp)
        tr4w.analyze(text, labels, candidate_pos=['NOUN', 'PROPN'], window_size=4, lower=False)
        return self.get_tags(tr4w.doc, tr4w.node_weight, num)

    def normalize_entity(self, ent):
        return re.sub(r'^the\s+', "", ent, flags=re.IGNORECASE)

    def get_tags(self, doc, node_weight, number=20):
        entity_weights, full_term_map = self._build_entity_weights(doc)
        tags = self._merge_terms(node_weight, entity_weights, full_term_map, number)
        self.tags = OrderedDict(sorted(tags.items(), key=lambda t: t[1], reverse=True))
        return self.tags

    def _build_entity_weights(self, doc):
        """Walk the spaCy entities and accumulate per-entity weights.

        Returns:
            entity_weights: Counter of normalized entity text -> accumulated weight.
            full_term_map: dict of lemma -> set(FullTerm) for multi-word entities,
                used later to promote single tokens to their full phrase.
        """
        entity_weights = Counter()
        occurrence = Counter()
        full_term_map = dict()

        for entity in doc.ents:
            weight = 1
            text = self.normalize_entity(entity.text)
            entity_doc = self.nlp(text)
            words = []
            token_count = len(entity_doc)

            for token in entity_doc:
                if token.pos_ not in self.config.exclude_pos:
                    words.append(token.text)

            text = " ".join(words)
            occurrence[text] += 1

            if token_count > 1:
                weight = self.entity_label_weights[entity.label_] if entity.label_ in self.entity_label_weights else 1
                for token in entity_doc:
                    if token.is_stop is False:
                        bonus_multiplier = self.config.full_term_weight_bonus[entity.label_] \
                            if entity.label_ in self.config.full_term_weight_bonus else 1

                        full_term = FullTerm(text, bonus_multiplier)
                        full_term_map.setdefault(token.lemma_, set())
                        full_term_map[token.lemma_].add(full_term)

            if self.config.multiple_occurrence_multiplier:
                weight = weight * occurrence[text]

            entity_weights[text] += weight

        return entity_weights, full_term_map

    def _resolve_ambiguity(self, full_terms, entity_weights):
        """Pick the winning full phrase when a lemma maps to several of them.

        Returns (is_ambiguity, ambiguity_winner). ambiguity_winner is None when the
        phrases are tied (no clear winner) or when ambiguity resolution is disabled.
        """
        is_ambiguity = False
        ambiguity_winner = None
        if len(full_terms) > 1 and self.remove_ambiguity is True:
            is_ambiguity = True
            phrase_counts = Counter()

            for full_term in full_terms:
                text = full_term.text
                phrase_counts[text] += entity_weights[text] or 1
            if phrase_counts.most_common()[0][1] != phrase_counts.most_common()[len(phrase_counts) - 1][1]:
                ambiguity_winner = phrase_counts.most_common()[0][0]
        return is_ambiguity, ambiguity_winner

    def _merge_terms(self, node_weight, entity_weights, full_term_map, number):
        """Combine TextRank scores with entity weights, promoting single tokens to
        their multi-word entity phrase where that scores higher."""
        tags = dict()
        node_weight = OrderedDict(sorted(node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (term, rank_score) in enumerate(node_weight.items()):
            # we are only interested in replacing the ones that have multi worded version
            if term in full_term_map:
                is_ambiguity, ambiguity_winner = self._resolve_ambiguity(full_term_map[term], entity_weights)

                single_weight = 0
                if term in entity_weights:
                    single_weight = entity_weights[term]

                if self.remove_ambiguity is True and is_ambiguity and ambiguity_winner is None:
                    tags[term] = tags.setdefault(term, 0) + rank_score
                    continue

                for full_term in full_term_map[term]:
                    text = full_term.text
                    weight_bonus = full_term.weight
                    if ambiguity_winner and ambiguity_winner != text:
                        continue

                    extra_weight = tags[text] if text in tags else 0
                    multi_weight = (entity_weights[text] + extra_weight) * weight_bonus

                    if single_weight >= multi_weight:
                        tags[term] = tags.setdefault(term, 0) + (rank_score * single_weight)
                    else:
                        tags[text] = tags.setdefault(text, 0) + (rank_score * multi_weight)

            else:
                weight = entity_weights[term]
                if weight > 0:
                    tags[term] = rank_score * weight

            if i > number:
                break

        return tags

    def to_json(self):
        return json.dumps(self.tags)


class FullTerm:
    def __init__(self, text, weight):
        self.text = text
        self.weight = weight

    def __eq__(self, other):
        return isinstance(other, FullTerm) and self.text == other.text

    def __hash__(self):
        return hash(self.text)

    def __str__(self):
        return f"{self.text}:{self.weight}"
