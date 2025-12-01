import gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import warnings
import os
import re
import preprocess
import spacy

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

#Limit each document to avoid spaCy max_length + huge memory use
MAX_CHARS = 200000  

trimmed_docs = []
for i, doc in enumerate(raw_docs):
    if len(doc) > MAX_CHARS:
        print(f"Trimming doc {i} from {len(doc)} to {MAX_CHARS} chars")
        trimmed_docs.append(doc[:MAX_CHARS])
    else:
        trimmed_docs.append(doc)

raw_docs = trimmed_docs

def resolve_coref_text(doc: spacy.tokens.Doc) -> str:
 
    tokens = [t.text for t in doc]
    for chain in doc._.coref_chains:
        # antecedent is the text of the first mention in the chain
        first_mention = chain[0]  
        antecedent = " ".join(doc[i].text for i in first_mention)

        # replace all subsequent mentions with the antecedent text
        for mention in list(chain)[1:]:
            if not mention:
                continue
            first_idx = mention[0]
            tokens[first_idx] = antecedent
            for j in mention[1:]:
                tokens[j] = ""

    resolved = " ".join(t for t in tokens if t)
    return resolved

nlp = spacy.load("en_core_web_lg")

def apply_coreferee_to_corpus(docs):
    resolved_docs = []
    for i, text in enumerate(docs):
        print(f"Applying pronoun resolution {i+1}/{len(docs)}", end="\r")
        doc = nlp(text)
        try:
            resolved = resolve_coref_text(doc)
        except Exception:
            resolved = text  
        resolved_docs.append(resolved)
    print() 
    return resolved_docs


def run_lda_pipeline(texts_tokenized, num_topics, html_filename, label):
    # Dictionary
    dictionary = corpora.Dictionary(texts_tokenized)
    corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

    print(f"\nTraining LDA {label}")
    print("Dictionary size:", len(dictionary))
    print("Corpus size:", len(corpus))

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        eta='auto'
    )

    print("\nTopics:")
    pprint(lda_model.print_topics())

    # Per-document topic distributions (optional)
    doc_topics = [lda_model.get_document_topics(bow) for bow in corpus]
    for i, topics in enumerate(doc_topics[:5]):  # show first 5 docs
        print(f"\nDocument {i} ({label}):")
        for topic_num, prob in topics:
            print(f"  Topic {topic_num}: {prob:.4f}")

    # Visualization
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, html_filename)
    print(f"\nVisualization saved to {html_filename}")

    return lda_model, corpus, dictionary

def preprocess_docs(docs):
    return [preprocess.preprocess(doc) for doc in docs]

print("\n Running LDA without pronoun resolution")
texts_no_coref = preprocess_docs(raw_docs)
lda_no_coref, corpus_no_coref, dict_no_coref = run_lda_pipeline(
    texts_tokenized=texts_no_coref,
    num_topics=5,
    html_filename="lda_visualization_no_resolution.html",
    label="no_coref"
)

print("\n Running LDA with pronoun resolution")
resolved_docs = apply_coreferee_to_corpus(raw_docs)
texts_with_coref = preprocess_docs(resolved_docs)

lda_with_coref, corpus_with_coref, dict_with_coref = run_lda_pipeline(
    texts_tokenized=texts_with_coref,
    num_topics=5,
    html_filename="lda_visualization_with_resolution.html",
    label="with_coref"
)