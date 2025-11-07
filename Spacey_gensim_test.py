import spacy, coreferee
from gensim.models import Word2Vec

print("spaCy version:", spacy.__version__)

nlp = spacy.load("en_core_web_trf")  # or en_core_web_lg
if not nlp.has_pipe("coreferee"):
    nlp.add_pipe("coreferee")

text = "Angela told Bob she would email him after she landed. He thanked her."
doc = nlp(text)

print("\nCoreference chains:")
for chain in doc._.coref_chains:
    # each `mention` is a list of token indices, e.g. [1] or [16, 19]
    mention_texts = []
    for mention in chain:
        # mention is a list of indices -> turn into a Span/string
        tokens = [doc[i].text for i in mention]
        mention_texts.append(" ".join(tokens))
    print(" → ".join(mention_texts))
    
# Gensim demo
sentences = [["hello", "world"], ["machine", "learning", "is", "fun"]]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=2)
print("\nMost similar to 'learning':", model.wv.most_similar("learning"))