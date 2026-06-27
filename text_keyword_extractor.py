import json
import re
from collections import OrderedDict, Counter, UserDict, UserList

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

excludePos = ["PUNCT", "PART"];
mwEntityLabelWeight = {"PERSON": 5, "ORG": 5, "NORP": 3, "GPE": 1.5, "EVENT": 5, "PRODUCT": 5, "WORK_OF_ART": 5,
                       "DATE": 0}
multipleOccuranceMultiplier = True
multiWinsMode = True
fullTermWeightBonusCoefficient = {"PERSON": 2}


class TextRank4Keyword():
    def __init__(self, nlp):
        self.nlp = nlp;
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.doc = None;
        self.keywords = None;

    def set_stopwords(self, stopwords):
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, labels, lower):
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
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

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, out=np.zeros_like(g), where=norm != 0)

        return g_norm

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            # print(key + ' - ' + str(value))
            if i > number:
                break

    def analyze(self, text,
                labels=[],
                candidate_pos=['NOUN', 'PROPN'],
                window_size=3, lower=False, stopwords=list()):

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spacy
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

        # Initialization for weight(pagerank value)
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
    def __init__(self, nlp):
        self.nlp = nlp;
        self.removeAmbiguity = True;
        self.tags = OrderedDict()
        self.mwEntityLabelWeight = mwEntityLabelWeight.copy();

    # labels: https://spacy.io/api/annotation#dependency-parsing
    def extract(self, text, labels, num=20):
        if isinstance(labels, dict):
            self.mwEntityLabelWeight.update(labels);
            labels = list(labels.keys());

        tr4w = TextRank4Keyword(self.nlp);
        tr4w.analyze(text, labels, candidate_pos=['NOUN', 'PROPN'], window_size=4, lower=False)
        tr4w.get_keywords(20);
        return self.get_tags(tr4w.doc, tr4w.node_weight, num)

    def normalize_entity(self, ent):
        return re.sub('^the\s+', "", ent, flags=re.IGNORECASE)

    def get_tags(self, doc, node_weight, number=20):
        cnt = Counter();
        occurance = Counter();
        mapFullTerm = UserDict()

        for entity in doc.ents:
            # weight = swEntityLabelWeight[entity.label_] if entity.label_ in swEntityLabelWeight else 1
            weight = 1
            txt = self.normalize_entity(entity.text)
            curDoc = self.nlp(txt)
            wordsAr = UserList()
            tokens = [token.text for token in curDoc];

            for token in curDoc:
                if token.pos_ not in excludePos:
                    wordsAr.append(token.text)

            txt = " ".join(wordsAr)
            occurance[txt] += 1;

            if len(tokens) > 1:
                weight = self.mwEntityLabelWeight[entity.label_] if entity.label_ in self.mwEntityLabelWeight else 1
                for token in curDoc:
                    if token.is_stop is False:
                        additionalBonusMultiplier = fullTermWeightBonusCoefficient[entity.label_] \
                            if entity.label_ in fullTermWeightBonusCoefficient else 1

                        fullTerm = FullTerm(txt, additionalBonusMultiplier);
                        mapFullTerm.setdefault(token.lemma_, set());
                        mapFullTerm[token.lemma_].add(fullTerm);

            if multipleOccuranceMultiplier:
                weight = weight * occurance[txt]

            cnt[txt] += weight

        tags = dict()
        node_weight = OrderedDict(sorted(node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            # we are only interested in replacing the ones that have multi worded version
            if key in mapFullTerm:
                isAmbiguity = False
                ambiguityWinner = None;
                if len(mapFullTerm[key]) > 1 and self.removeAmbiguity is True:
                    isAmbiguity = True
                    pCnt = Counter()

                    for fullTerm in mapFullTerm[key]:
                        txt = fullTerm.txt
                        pCnt[txt] += cnt[txt] or 1;
                    if pCnt.most_common()[0][1] != pCnt.most_common()[len(pCnt) - 1][1]:
                        ambiguityWinner = pCnt.most_common()[0][0]

                singleWeight = 0
                if key in cnt:
                    singleWeight = cnt[key]

                if self.removeAmbiguity is True and isAmbiguity and ambiguityWinner is None:
                    tags[key] = tags.setdefault(key, 0) + value
                    continue

                for fullTerm in mapFullTerm[key]:
                    txt = fullTerm.txt
                    weightBonus = fullTerm.weight
                    if ambiguityWinner and ambiguityWinner != txt:
                        continue

                    extraWeight = tags[txt] if txt in tags else 0
                    multiWeight = (cnt[txt] + extraWeight) * weightBonus

                    # if multiWinsMode or multiWeight > singleWeight:
                    #     tags[txt] = tags.setdefault(txt, 0) + value;
                    # else:
                    #     tags[key] = tags.setdefault(key, 0) + value;

                    if singleWeight >= multiWeight:
                        tags[key] = tags.setdefault(key, 0) + (value * singleWeight)
                    else:
                        tags[txt] = tags.setdefault(txt, 0) + (value * multiWeight)


            else:
                weight = cnt[key]
                if weight > 0:
                    tags[key] = value * weight

            if i > number:
                break

        self.tags = OrderedDict(sorted(tags.items(), key=lambda t: t[1], reverse=True));
        return self.tags;

    def to_json(self):
        jsonObj = json.dumps(self.tags);
        return jsonObj;


class FullTerm:
    def __init__(self, txt, weight):
        self.txt = txt
        self.weight = weight

    def __eq__(self, other):
        return isinstance(other, FullTerm) and self.txt == other.txt

    def __hash__(self):
        return hash(self.txt)

    def __str__(self):
        return f"{self.txt}:{self.weight}"
