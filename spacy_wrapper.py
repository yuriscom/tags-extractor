import spacy_module_loader


class SpacyWrapper:
    def __init__(self):
        self.nlp = None

    def init(self):
        print("LOADING NLP")
        self.nlp = spacy_module_loader.init_nlp()
        return self.nlp
