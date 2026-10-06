import numpy as np


def build_cooccurrence_matrix(documents, window_size=2):
    """Builds a word-word co-occurrence matrix within a sliding window."""
    tokenized_docs = [doc.lower().split() for doc in documents]
    vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))
    vocab_size = len(vocab)
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    co_matrix = np.zeros((vocab_size, vocab_size), dtype=float)

    for doc in tokenized_docs:
        for i, target in enumerate(doc):
            target_idx = word_to_idx[target]
            start = max(0, i - window_size)
            end = min(len(doc), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_idx = word_to_idx[doc[j]]
                    co_matrix[target_idx, context_idx] += 1.0

    return co_matrix, vocab, word_to_idx


def compute_ppmi_matrix(co_matrix):
    """
    Computes Positive Pointwise Mutual Information (PPMI) matrix:
    PPMI(w, c) = max(0, log2( P(w, c) / (P(w) * P(c)) ))
    """
    total_count = np.sum(co_matrix)
    row_sums = np.sum(co_matrix, axis=1)  # P(w) * total_count
    col_sums = np.sum(co_matrix, axis=0)  # P(c) * total_count

    ppmi_matrix = np.zeros_like(co_matrix)

    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            if co_matrix[i, j] > 0:
                p_wc = co_matrix[i, j] / total_count
                p_w = row_sums[i] / total_count
                p_c = col_sums[j] / total_count

                pmi = np.log2(p_wc / (p_w * p_c))
                ppmi_matrix[i, j] = max(0.0, pmi)

    return ppmi_matrix


def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two 1D vectors."""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_document_vector(doc, vocab_map, ppmi_matrix):
    """Averages word PPMI vectors to represent a document."""
    words = doc.lower().split()
    word_vecs = [
        ppmi_matrix[vocab_map[word]] for word in words if word in vocab_map
    ]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(ppmi_matrix.shape[1])


if __name__ == "__main__":
    # Sample Training Documents
    documents = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barked at the lazy fox",
        "quick brown dogs are fast and smart",
    ]

    # 1. Build Co-occurrence Matrix & Compute PPMI
    window_size = 2
    co_matrix, vocab, vocab_map = build_cooccurrence_matrix(
        documents, window_size=window_size
    )
    ppmi_matrix = compute_ppmi_matrix(co_matrix)

    print("=" * 65)
    print(" PPMI MATRIX & COSINE SIMILARITY")
    print("=" * 65)

    print(f"\nVocabulary ({len(vocab)} words):")
    print(vocab)

    print(
        f"\nPPMI Matrix Excerpt (Shape: {ppmi_matrix.shape[0]}x{ppmi_matrix.shape[1]}):"
    )
    print(np.round(ppmi_matrix[:5, :5], 4))

    # 2. Word Cosine Similarity
    word1, word2 = "fox", "dog"
    v1 = ppmi_matrix[vocab_map[word1]]
    v2 = ppmi_matrix[vocab_map[word2]]
    word_sim = cosine_similarity(v1, v2)

    print("\n" + "-" * 65)
    print(" WORD SIMILARITY")
    print("-" * 65)
    print(f"Word 1: '{word1}'")
    print(f"Word 2: '{word2}'")
    print(
        f"Cosine Similarity ('{word1}' vs '{word2}'): {word_sim:.4f}"
    )

    # 3. Document Cosine Similarity (using averaged word PPMI embeddings)
    doc_idx1, doc_idx2 = 0, 1
    doc_vec1 = get_document_vector(documents[doc_idx1], vocab_map, ppmi_matrix)
    doc_vec2 = get_document_vector(documents[doc_idx2], vocab_map, ppmi_matrix)
    doc_sim = cosine_similarity(doc_vec1, doc_vec2)

    print("\n" + "-" * 65)
    print(" DOCUMENT SIMILARITY")
    print("-" * 65)
    print(f"Doc 1: '{documents[doc_idx1]}'")
    print(f"Doc 2: '{documents[doc_idx2]}'")
    print(
        f"Cosine Similarity (Doc 1 vs Doc 2): {doc_sim:.4f}"
    )
    print("=" * 65)
