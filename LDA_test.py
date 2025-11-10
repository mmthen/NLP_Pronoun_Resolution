import gensim
from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Example text corpus (list of documents, each document = list of tokens)
texts = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "is", "a", "branch", "of", "machine", "learning"],
    ["artificial", "intelligence", "and", "machine", "learning", "are", "related"],
    ["natural", "language", "processing", "is", "a", "field", "of", "ai"],
    ["computer", "vision", "is", "used", "in", "ai", "applications"]
]

# Create a dictionary mapping for all words
dictionary = corpora.Dictionary(texts)

# Convert the tokenized documents into bag-of-words representation
corpus = [dictionary.doc2bow(text) for text in texts]

print("Dictionary tokens:", dictionary.token2id)
print("Corpus:", corpus)

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,        # number of topics you want to extract
    random_state=42,
    passes=10,           # number of passes through the corpus
    alpha='auto',
    eta='auto'
)

# Print out topics
pprint(lda_model.print_topics())

new_doc = ["deep", "learning", "improves", "machine", "vision"]
new_bow = dictionary.doc2bow(new_doc)
topic_distribution = lda_model.get_document_topics(new_bow)

print("\nTopic distribution for new document:")
pprint(topic_distribution)

vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, "lda_visualization.html")
print("✅ Visualization saved to lda_visualization.html — open it in your browser!")