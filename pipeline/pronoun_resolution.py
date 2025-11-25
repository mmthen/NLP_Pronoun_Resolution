# pipeline/pronoun_resolution.py
import spacy
import coreferee


def load_coref_model():
    print("Loading spaCy transformer + Coreferee...")
    nlp = spacy.load("en_core_web_trf")
    if "coreferee" not in nlp.pipe_names:
        nlp.add_pipe("coreferee")
    return nlp


def resolve_pronouns(docs, nlp):
    resolved_docs = []

    for text in docs:
        doc = nlp(text)
        tokens = []

        for token in doc:
            reps = doc._.coref_chains.resolve(token)
            if reps:
                tokens.append(" ".join(t.text for t in reps))
            else:
                tokens.append(token.text)

        resolved_docs.append(" ".join(tokens))
    return resolved_docs
