import os
import spacy
from spacy.symbols import ORTH
import spacy_module_loader

### uncomment if decide to use config
# import configparser
# config = configparser.ConfigParser()
# config.read('config.ini')

lemma_fix_map = {"noun|data": "data"}

class SpacyWrapper():
    def __init__(self):
        self.nlp = None

    def init(self):
        ### uncomment if decide to use config
        # nlp = spacy.load(config.get("spacy", "dictionary"))
        print("LOADING NLP");
        nlp = spacy_module_loader.init_nlp()

        # nlp.tokenizer.add_special_case("gimme", [{ORTH: "give"}, {ORTH: "me"}])

        # table = nlp.vocab.lookups.get_table("lemma_exc")
        # for key in lemma_fix_map:
        #     key_lst = key.split("|");
        #     table[key_lst[0]][key_lst[1]] = [lemma_fix_map[key]]


        spacy.explain("NORP")
        return nlp;
