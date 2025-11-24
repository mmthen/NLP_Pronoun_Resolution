import gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import warnings
import os
import re

from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint
from gensim.parsing.preprocessing import STOPWORDS

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def load_all_documents(root="data/raw"):
    docs = []
    for topic in os.listdir(root):
        topic_folder = os.path.join(root, topic)
        if not os.path.isdir(topic_folder):
            continue
        for fname in os.listdir(topic_folder):
            if fname.endswith(".txt"):
                with open(os.path.join(topic_folder, fname), encoding="utf-8") as f:
                    docs.append(f.read())
    return docs

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    return tokens

raw_docs = load_all_documents()
texts = [tokenize(doc) for doc in raw_docs]
# Create a dictionary mapping for all words
dictionary = corpora.Dictionary(texts)

# Convert the tokenized documents into bag-of-words representation
corpus = [dictionary.doc2bow(text) for text in texts]

print("Dictionary tokens:", dictionary.token2id)
print("Corpus:", corpus)

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,        # number of topics you want to extract
    random_state=42,
    passes=10,           # number of passes through the corpus
    alpha='auto',
    eta='auto'
)

# Print out topics
pprint(lda_model.print_topics())

doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]

for i, topics in enumerate(doc_topics):
    print(f"Document {i}:")
    for topic_num, prob in topics:
        print(f"  Topic {topic_num}: {prob:.4f}")
    print()

#Help visualize the model
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, "lda_visualization_no_resoultion.html")
print("Visualization saved to lda_visualization_no_resoultion")