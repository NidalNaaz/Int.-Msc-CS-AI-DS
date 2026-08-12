import re
import math
from collections import Counter


# =========================================================
# 1. TOKENIZATION
# =========================================================

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


# =========================================================
# 2. BUILD CORPUS MODEL
# =========================================================

def build_model(corpus):
    words = tokenize(corpus)

    vocabulary = set(words)
    unigram_counts = Counter(words)

    total_words = len(words)

    return vocabulary, unigram_counts, total_words


# =========================================================
# 3. GENERATE EDIT-DISTANCE-1 CANDIDATES
# =========================================================

def generate_candidates(word, vocabulary):

    candidates = set()
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    # -----------------------------------------------------
    # Deletion
    # -----------------------------------------------------

    for i in range(len(word)):

        candidate = word[:i] + word[i + 1:]

        if candidate in vocabulary:
            candidates.add(candidate)

    # -----------------------------------------------------
    # Insertion
    # -----------------------------------------------------

    for i in range(len(word) + 1):

        for char in alphabet:

            candidate = (
                word[:i] +
                char +
                word[i:]
            )

            if candidate in vocabulary:
                candidates.add(candidate)

    # -----------------------------------------------------
    # Substitution
    # -----------------------------------------------------

    for i in range(len(word)):

        for char in alphabet:

            if char == word[i]:
                continue

            candidate = (
                word[:i] +
                char +
                word[i + 1:]
            )

            if candidate in vocabulary:
                candidates.add(candidate)

    # -----------------------------------------------------
    # Transposition
    # -----------------------------------------------------

    for i in range(len(word) - 1):

        candidate = (
            word[:i]
            + word[i + 1]
            + word[i]
            + word[i + 2:]
        )

        if candidate in vocabulary:
            candidates.add(candidate)

    return candidates


# =========================================================
# 4. MINIMUM EDIT DISTANCE
# =========================================================

def edit_distance(a, b):

    m = len(a)
    n = len(b)

    dp = [[0] * (n + 1)
          for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n]


# =========================================================
# 5. CHANNEL PROBABILITY
# =========================================================

def channel_probability(typo, candidate):

    distance = edit_distance(
        candidate,
        typo
    )

    # Candidate should normally be
    # one edit away.

    if distance == 0:
        return 0.90

    if distance == 1:
        return 0.10

    return 0.001


# =========================================================
# 6. PRIOR PROBABILITY
# =========================================================

def word_probability(word,
                     unigram_counts,
                     total_words,
                     vocabulary_size):

    # Laplace smoothing

    probability = (
        unigram_counts[word] + 1
    ) / (
        total_words + vocabulary_size
    )

    return probability


# =========================================================
# 7. NOISY CHANNEL SCORE
# =========================================================

def noisy_channel_score(
        typo,
        candidate,
        unigram_counts,
        total_words,
        vocabulary_size):

    prior = word_probability(
        candidate,
        unigram_counts,
        total_words,
        vocabulary_size
    )

    channel = channel_probability(
        typo,
        candidate
    )

    return prior * channel


# =========================================================
# 8. SPELL CHECK
# =========================================================

def spell_check(
        sentence,
        vocabulary,
        unigram_counts,
        total_words):

    words = tokenize(sentence)

    corrected_words = words.copy()

    results = []

    vocabulary_size = len(vocabulary)

    for i, word in enumerate(words):

        # Correct word
        if word in vocabulary:
            continue

        candidates = generate_candidates(
            word,
            vocabulary
        )

        if not candidates:

            results.append(
                (word, {}, None)
            )

            continue

        scores = {}

        for candidate in candidates:

            score = noisy_channel_score(
                word,
                candidate,
                unigram_counts,
                total_words,
                vocabulary_size
            )

            scores[candidate] = score

        best_candidate = max(
            scores,
            key=scores.get
        )

        corrected_words[i] = best_candidate

        results.append(
            (
                word,
                scores,
                best_candidate
            )
        )

    return corrected_words, results


# =========================================================
# 9. DISPLAY RESULTS
# =========================================================

def display_results(results):

    print("\n" + "=" * 60)
    print("NOISY CHANNEL SPELL CHECK RESULTS")
    print("=" * 60)

    if not results:

        print("\nNo spelling errors detected.")

        return

    for word, scores, best in results:

        print(
            f"\nMisspelled word: {word}"
        )

        if not scores:

            print(
                "No edit-distance-1 candidates found."
            )

            continue

        print("\nCandidates:")

        for candidate, score in sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True):

            print(
                f"  {candidate:<15}"
                f"Score = {score:.10f}"
            )

        print(
            f"\nBest candidate: {best}"
        )


# =========================================================
# 10. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("NOISY CHANNEL SPELL CHECKER")
    print("=" * 60)

    # -----------------------------------------------------
    # Demonstration corpus
    # -----------------------------------------------------

    corpus = """
    the cat sat on the mat
    the cat ate the food
    the dog sat on the mat
    the dog ate the food
    the boy sat on the chair
    the boy ate the food
    the girl sat on the chair
    the girl ate the food
    the cat chased the dog
    the dog chased the cat
    the boy played with the dog
    the girl played with the cat
    """

    vocabulary, unigram_counts, total_words = \
        build_model(corpus)

    print(
        "\nVocabulary size:",
        len(vocabulary)
    )

    print(
        "Total corpus words:",
        total_words
    )

    sentence = input(
        "\nEnter a sentence containing "
        "possible spelling errors:\n"
    )

    corrected_words, results = spell_check(
        sentence,
        vocabulary,
        unigram_counts,
        total_words
    )

    display_results(results)

    corrected_sentence = " ".join(
        corrected_words
    )

    print("\n" + "=" * 60)
    print("CORRECTED SENTENCE")
    print("=" * 60)

    print(corrected_sentence)


if __name__ == "__main__":
    main()
