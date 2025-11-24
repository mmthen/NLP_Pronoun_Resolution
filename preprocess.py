import string
import spacy

# spaCy preprocessing pipline
nlp = spacy.load('en_core_web_lg', disable=["parser", "ner", "textcat", "coreferee"])

PUNCT = set(string.punctuation)

