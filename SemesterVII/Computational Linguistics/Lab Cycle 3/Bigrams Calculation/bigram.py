from collections import Counter, defaultdict


def build_bigram_model(corpus):
    """
    Builds unigram and bigram frequency counts from a corpus.
    Applies start (<s>) and end (</s>) tokens to each sentence.
    """
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)

    for sentence in corpus:
        tokens = ["<s>"] + sentence.lower().split() + ["</s>"]

        for word in tokens:
            unigram_counts[word] += 1

        for w1, w2 in zip(tokens[:-1], tokens[1:]):
            bigram_counts[w1][w2] += 1

    return unigram_counts, bigram_counts


def calculate_sentence_probability(
    sentence, unigram_counts, bigram_counts, use_laplace=True
):
    """
    Calculates the probability of a sentence using the bigram model.
    Applies Laplace smoothing if requested to avoid zero-probability issues.
    """
    tokens = ["<s>"] + sentence.lower().split() + ["</s>"]
    vocabulary = set(unigram_counts.keys())
    vocab_size = len(vocabulary)

    probability = 1.0
    bigram_details = []

    for w1, w2 in zip(tokens[:-1], tokens[1:]):
        count_w1_w2 = bigram_counts[w1][w2]
        count_w1 = unigram_counts[w1]

        if use_laplace:
            # P(w2|w1) with Laplace (add-1) smoothing: (count(w1, w2) + 1) / (count(w1) + |V|)
            p_bigram = (count_w1_w2 + 1) / (count_w1 + vocab_size)
        else:
            # Unsmoothed MLE probability
            p_bigram = count_w1_w2 / count_w1 if count_w1 > 0 else 0.0

        probability *= p_bigram
        bigram_details.append((w1, w2, count_w1_w2, count_w1, p_bigram))

    return probability, bigram_details


if __name__ == "__main__":
    # Sample Training Corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barked at the lazy fox",
        "quick brown dogs are fast and smart",
    ]

    # Build model
    unigram_counts, bigram_counts = build_bigram_model(corpus)

    # Test Sentences
    test_sentences = [
        "the lazy dog",
        "the quick brown fox",
    ]

    print("=" * 65)
    print(" BIGRAM LANGUAGE MODEL & SENTENCE PROBABILITY")
    print("=" * 65)

    for test_sentence in test_sentences:
        prob, details = calculate_sentence_probability(
            test_sentence, unigram_counts, bigram_counts, use_laplace=True
        )

        print(f"\nTest Sentence : '{test_sentence}'")
        print("-" * 65)
        print(f"{'Bigram (w1 -> w2)':<25} | {'Count(w1,w2)':<12} | {'P(w2|w1)':<10}")
        print("-" * 65)

        for w1, w2, count_bigram, count_unigram, p_val in details:
            print(f"{w1 + ' -> ' + w2:<25} | {count_bigram:<12} | {p_val:.5f}")

        print("-" * 65)
        print(f"Total Sentence Probability: {prob:.10e}\n")
