import gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import warnings
import os
import re
import preprocess

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

raw_docs = load_all_documents()
texts = [preprocess(doc) for doc in raw_docs]

# Create a dictionary mapping for all words
dictionary = corpora.Dictionary(texts)

# Convert the tokenized documents into bag-of-words representation
corpus = [dictionary.doc2bow(text) for text in texts]

# Build LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,        
    random_state=42,
    passes=10,           
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