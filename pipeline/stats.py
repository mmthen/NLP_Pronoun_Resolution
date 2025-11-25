import numpy as np
from scipy.spatial.distance import jensenshannon

"""
Computes the Jensen–Shannon Divergence between:

    - the topic distribution of each original document

    - the topic distribution of the pronoun-resolved 
      version of that same document

How much does pronoun resolution change what topic the document belongs to?
"""


# Convert sparse topic distribution a numpy vector
def dist_to_vec(dist, num_topics):
    """
    dist = list of (topic_id, prob)
    returns a dense vector of length num_topics
    """
    vec = np.zeros(num_topics, dtype=float)
    for topic_id, prob in dist:
        vec[topic_id] = prob
    return vec


def compute_divergence(lda_orig, corpus_orig, lda_res, corpus_res, num_topics=5):
    orig_vecs = []
    res_vecs = []

    for doc in corpus_orig:
        dist = lda_orig.get_document_topics(doc, minimum_probability=0)
        orig_vecs.append(dist_to_vec(dist, num_topics))

    for doc in corpus_res:
        dist = lda_res.get_document_topics(doc, minimum_probability=0)
        res_vecs.append(dist_to_vec(dist, num_topics))

        # Normalize to avoid all-zero vectors
        def normalize(v):
            s = v.sum()
            if s == 0:
                return np.ones_like(v) / len(v)  # uniform distribution fallback
            return v / s

        orig_vecs = [normalize(v) for v in orig_vecs]
        res_vecs = [normalize(v) for v in res_vecs]

        distances = [
            jensenshannon(v1, v2)
            for v1, v2 in zip(orig_vecs, res_vecs)
        ]

        print("\n===== TOPIC SHIFT (Jensen–Shannon Divergence) =====")
        print(f"Mean: {np.mean(distances):.4f}")
        print(f"Min : {np.min(distances):.4f}")
        print(f"Max : {np.max(distances):.4f}")

        return distances
