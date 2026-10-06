import math
import numpy as np


def compute_tfidf_matrix(documents):
    """Computes the TF-IDF matrix and vocabulary for a given list of document strings."""
    # Tokenize corpus
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Build unique vocabulary
    vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))
    num_docs = len(documents)

    # Document Frequency (DF) for each term
    df = {term: sum(1 for doc in tokenized_docs if term in doc) for term in vocab}

    # Inverse Document Frequency (IDF)
    # Formula: log(N / DF(t))
    idf = {term: math.log(num_docs / df[term]) for term in vocab}

    # Term Frequency (TF) matrix and TF-IDF matrix
    tfidf_matrix = []
    for doc in tokenized_docs:
        doc_len = len(doc)
        # Term Frequency: count(t, d) / total_words(d)
        doc_tfidf = [
            (doc.count(term) / doc_len) * idf[term] for term in vocab
        ]
        tfidf_matrix.append(doc_tfidf)

    return np.array(tfidf_matrix), vocab


def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two 1D vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    # Sample Training Corpus
    documents = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barked at the lazy fox",
        "quick brown dogs are fast and smart",
    ]

    # 1. Compute TF-IDF Matrix
    tfidf_matrix, vocab = compute_tfidf_matrix(documents)

    print("=" * 65)
    print(" TF-IDF MATRIX & COSINE SIMILARITY")
    print("=" * 65)

    print(f"\nVocabulary ({len(vocab)} words):")
    print(vocab)

    print("\nTF-IDF Matrix (Shape: Documents x Words):")
    print(np.round(tfidf_matrix, 4))

    # 2. Document Cosine Similarity
    doc_idx1, doc_idx2 = 0, 1
    doc_sim = cosine_similarity(
        tfidf_matrix[doc_idx1], tfidf_matrix[doc_idx2]
    )

    print("\n" + "-" * 65)
    print(" DOCUMENT SIMILARITY")
    print("-" * 65)
    print(f"Doc 1: '{documents[doc_idx1]}'")
    print(f"Doc 2: '{documents[doc_idx2]}'")
    print(
        f"Cosine Similarity (Doc 1 vs Doc 2): {doc_sim:.4f}"
    )

    # 3. Word Cosine Similarity (using columns of TF-IDF matrix as word vectors)
    word1, word2 = "fox", "dog"
    w1_idx = vocab.index(word1)
    w2_idx = vocab.index(word2)

    # Extract column vectors for words across all documents
    word_vec1 = tfidf_matrix[:, w1_idx]
    word_vec2 = tfidf_matrix[:, w2_idx]

    word_sim = cosine_similarity(word_vec1, word_vec2)

    print("\n" + "-" * 65)
    print(" WORD SIMILARITY")
    print("-" * 65)
    print(f"Word 1: '{word1}' -> Vector: {np.round(word_vec1, 4)}")
    print(f"Word 2: '{word2}' -> Vector: {np.round(word_vec2, 4)}")
    print(
        f"Cosine Similarity ('{word1}' vs '{word2}'): {word_sim:.4f}"
    )
    print("=" * 65)
