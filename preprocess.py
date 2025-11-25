# preprocess.py
import spacy
import string

nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat", "coreferee"])
PUNCT = set(string.punctuation)


def preprocess(text):
    """
    Convert raw text into clean tokens for LDA:
    - lowercase
    - remove stopwords
    - remove numbers
    - remove punctuation
    - alphabetic tokens only
    - lemmatize
    """
    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_stop:
            continue
        if token.is_punct or token.text in PUNCT:
            continue
        if token.like_num:
            continue

        lemma = token.lemma_.strip()
        if len(lemma) <= 2:
            continue
        if not lemma.isalpha():
            continue

        tokens.append(lemma)

    return tokens


if __name__ == "__main__":
    test = "Angela was talking to Bob after she landed at 5pm."
    print(preprocess(test))
