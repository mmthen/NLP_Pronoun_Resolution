from gensim import corpora, models


def train_lda(tokenized_docs, num_topics=5):
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]

    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha='symmetric',
        eta='symmetric',
        passes=20,
        random_state=42
    )

    return lda, dictionary, corpus


TOPIC_LABELS = {
    0: "Politics",
    1: "Entertainment",
    2: "Sports",
    3: "Tech",
    4: "Travel"
}


def print_topics(label, lda_model, num_words=20):
    print(f"\n===== {label} =====")

    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

    for topic_id, word_probs in topics:
        # top 3 keywords
        top_words = [w for w, p in word_probs[:3]]
        short_label = ", ".join(top_words)

        topic_name = TOPIC_LABELS.get(topic_id, "Unknown")

        print(f"\nTopic {topic_id}  →  {topic_name}  |  Top words: {short_label}")
        print(" ".join([f"{w}({p:.3f})" for w, p in word_probs]))
