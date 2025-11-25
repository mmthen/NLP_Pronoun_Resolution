from load_documents import load_documents
from pronoun_resolution import load_coref_model, resolve_pronouns
from lda_modeling import train_lda, print_topics
from stats import compute_divergence
from visualizations import (
    save_pyldavis,
    save_compare_html,
    save_topic_words_html
)

from preprocess import preprocess

"""
Run and process pipeline methods
"""

NUM_TOPICS = 5


def main():
    # Load documents
    documents, filenames = load_documents()

    # Preprocess
    pre_orig = [preprocess(d) for d in documents]

    nlp = load_coref_model()

    # Pronoun resolution
    resolved_docs = resolve_pronouns(documents, nlp)

    # preprocess resolved docs
    process_resolved = [preprocess(d) for d in resolved_docs]

    # Train LDAs
    lda_orig, dict_orig, corpus_orig = train_lda(pre_orig, NUM_TOPICS)
    lda_res, dict_res, corpus_res = train_lda(process_resolved, NUM_TOPICS)

    print_topics("BEFORE RESOLUTION", lda_orig)
    print_topics("AFTER RESOLUTION", lda_res)

    # Divergence computation
    compute_divergence(lda_orig, corpus_orig, lda_res, corpus_res)

    # Visualization
    save_pyldavis(lda_orig, corpus_orig, dict_orig, "lda_original.html")
    save_pyldavis(lda_res, corpus_res, dict_res, "lda_resolved.html")

    save_topic_words_html(lda_orig, "lda_original_top20.html",
                          "Top 20 Words Before Pronoun Resolution")

    save_topic_words_html(lda_res, "lda_resolved_top20.html",
                          "Top 20 Words After Pronoun Resolution")

    save_compare_html()

    print("Pipeline complete.")


if __name__ == '__main__':
    main()
