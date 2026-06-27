from collections import OrderedDict, Counter, UserDict, UserList
import numpy as np
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

# text = '''The Wandering Earth, described as China’s first big-budget science fiction thriller, quietly made it onto screens at AMC theaters in North America this weekend, and it shows a new side of Chinese filmmaking — one focused toward futuristic spectacles rather than China’s traditionally grand, massive historical epics. At the same time, The Wandering Earth feels like a throwback to a few familiar eras of American filmmaking. While the film’s cast, setting, and tone are all Chinese, longtime science fiction fans are going to see a lot on the screen that reminds them of other movies, for better or worse.'''
text = '''WASHINGTON - President Trump on Monday ripped Joe Biden's denial of a decades-old sexual assault allegation by a former Senate staffer - and said that Republicans get treated far more harshly than Democrats when accused of sexual misconduct.

'I mean, his choice of words weren't very good when he, you know, dismissed the allegation,' Trump said of Biden during an interview with The Post in the Oval Office.

'But he's got to fight his own battles, and we'll see how he does.'

Biden, the presumptive Democratic presidential nominee, flatly denied former aide Tara Reade's claim during a Friday interview with 'Morning Joe' on MSNBC, saying the alleged 1993 assault 'never happened,' while also refusing to open his Senate records held by the University of Delaware, citing past speeches he's made and engagements with foreign leaders.

Trump did not specify what part of Biden's denial he found fault with, but said he believes that Republicans face far harsher scrutiny when accused of sexual misconduct.

During his 2018 confirmation hearings, Supreme Court Justice Brett Kavanaugh 'was treated more unfairly than any human being I've ever seen. I've never seen anything like,' Trump said.

'He's a fine man with a fine family,' Trump said 'And there's never been a human being in the history of Congress that was treated worse than him.'

Kavanaugh faced misconduct allegations dating to his teen and college years from Stanford University psychology professor Christine Blasey Ford, Yale University classmate Deborah Ramirez and other women, some of whom later recanted.

'And now in the meantime three of the women have recanted,' Trump said.

'And the other one I don't believe her for a moment, which is Blasey Ford. Because the story doesn't check out. Even her friends and her family didn't check out. Everything didn't check out including the door. You know, the double door, right? You know what I mean.'

He added: 'Nothing checked out. And there's ever been a human being who has been treated more unfairly [than] a very fine man named Justice Kavanaugh.'

On Friday, Trump weighed in on Reade's claims during a radio interview with conservative pundit Dan Bongino, saying that he had personally faced 'nonsense false accusations' from women accusing him of misconduct and assault.

But Trump told Bongino, 'I watched Tara and she seems very credible.'''

# text = '''The arctic fox is an incredibly hardy animal that can survive frigid Arctic temperatures as low as –58°F in the treeless lands where it makes its home. It has furry soles, short ears, and a short muzzle—all-important adaptations to the chilly clime. Arctic foxes live in burrows, and in a blizzard they may tunnel into the snow to create shelter.'''
# text = '''Israel, formally known as the State of Israel, is a country in Western Asia, located on the southeastern shore of the Mediterranean Sea and the northern shore of the Red Sea. It has land borders with Lebanon to the north, Syria to the northeast, Jordan on the east, the Palestinian territories of the West Bank and Gaza Strip to the east and west, respectively, and Egypt to the southwest. The country contains geographically diverse features within its relatively small area. Israel's economic and technological center is Tel Aviv, while its seat of government and proclaimed capital is Jerusalem, although the state's sovereignty over Jerusalem has only partial recognition.'''

excludePos = ["PUNCT", "PART"];


class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.doc = None;
        self.keywords = None;
        self.removeAmbiguity = True;

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
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
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        self.doc = doc

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

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

    def normalizeEntity(self, ent):
        return re.sub('^the\s+', "", ent, flags=re.IGNORECASE)



    def getTags(self, number=20):
        cnt = Counter();
        mapMulti = UserDict()
        for entity in self.doc.ents:
            txt = self.normalizeEntity(entity.text)
            curDoc = nlp(txt)
            wordsAr = UserList()
            tokens = [token.text for token in curDoc];

            for token in curDoc:
                if token.pos_ not in excludePos:
                    wordsAr.append(token.text)

            txt = " ".join(wordsAr)

            if len(tokens) > 1:
                for token in curDoc:
                    if token.is_stop is False:
                        # mapMulti.setdefault(token.lower_, []).append(txt)
                        mapMulti.setdefault(token.text, []).append(txt)
            # cnt[txt.lower()] += 1
            cnt[txt] += 1

        tags = UserDict()
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            # lkey = key.lower();
            # we are only interested in replacing the ones that have multi worded version
            if key in mapMulti:
                isAmbiguity = False
                if len(mapMulti[key]) > 1 and self.removeAmbiguity is True:
                    pCnt = Counter()
                    for txt in mapMulti[key]:
                        pCnt[txt] += 1
                    if pCnt[0] == pCnt[1]:
                        isAmbiguity = True

                singleWeight = 0
                if key in cnt:
                    singleWeight = cnt[key]

                if self.removeAmbiguity is True and isAmbiguity:
                    tags[key] = tags.setdefault(key, 0) + value;
                    continue

                for txt in mapMulti[key]:
                    multiWeight = cnt[txt]
                    if singleWeight >= multiWeight:
                        tags[key] = tags.setdefault(key, 0) + value;
                    else:
                        tags[txt] = tags.setdefault(txt, 0) + value;
            else:
                tags[key] = value;

            if i > number:
                break

        print(tags);



tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos=['NOUN', 'PROPN'], window_size=4, lower=False)
# tr4w.get_keywords(20)
tr4w.getTags(20)
