import re
from collections import Counter


# ---------------------------------------------------------
# 1. TOKENIZATION
# ---------------------------------------------------------

def tokenize(text):
    """
    Convert text into lowercase alphabetic word tokens.
    """
    return re.findall(r"[a-z]+", text.lower())


# ---------------------------------------------------------
# 2. CREATE VOCABULARY AND BIGRAM FREQUENCIES
# ---------------------------------------------------------

def build_language_model(corpus):
    tokens = tokenize(corpus)

    vocabulary = set(tokens)

    unigram_counts = Counter(tokens)

    bigram_counts = Counter(
        zip(tokens[:-1], tokens[1:])
    )

    return vocabulary, unigram_counts, bigram_counts


# ---------------------------------------------------------
# 3. EDIT DISTANCE = 1 CANDIDATES
# ---------------------------------------------------------

def edit_distance_one(word, vocabulary):
    """
    Return vocabulary words that are exactly one edit
    away from the given word.
    """

    candidates = set()

    # -------------------------
    # Deletions
    # -------------------------
    for i in range(len(word)):
        candidate = word[:i] + word[i + 1:]

        if candidate in vocabulary:
            candidates.add(candidate)

    # -------------------------
    # Insertions
    # -------------------------
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for i in range(len(word) + 1):
        for char in alphabet:
            candidate = word[:i] + char + word[i:]

            if candidate in vocabulary:
                candidates.add(candidate)

    # -------------------------
    # Substitutions
    # -------------------------
    for i in range(len(word)):
        for char in alphabet:

            if char == word[i]:
                continue

            candidate = word[:i] + char + word[i + 1:]

            if candidate in vocabulary:
                candidates.add(candidate)

    # -------------------------
    # Transpositions
    # -------------------------
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


# ---------------------------------------------------------
# 4. BIGRAM PROBABILITY
# ---------------------------------------------------------

def bigram_probability(previous_word,
                       current_word,
                       unigram_counts,
                       bigram_counts):
    """
    Calculate P(current_word | previous_word)
    using an unsmoothed bigram model.
    """

    bigram_count = bigram_counts[
        (previous_word, current_word)
    ]

    previous_count = unigram_counts[previous_word]

    if previous_count == 0:
        return 0

    return bigram_count / previous_count


# ---------------------------------------------------------
# 5. SCORE CANDIDATE
# ---------------------------------------------------------

def candidate_score(previous_word,
                    candidate,
                    next_word,
                    unigram_counts,
                    bigram_counts):
    """
    Score a candidate using the surrounding bigrams.
    """

    score = 1.0

    # P(candidate | previous word)
    if previous_word:
        score *= bigram_probability(
            previous_word,
            candidate,
            unigram_counts,
            bigram_counts
        )

    # P(next word | candidate)
    if next_word:
        score *= bigram_probability(
            candidate,
            next_word,
            unigram_counts,
            bigram_counts
        )

    # Small unigram component to break ties
    score *= (
        unigram_counts[candidate] + 1
    )

    return score


# ---------------------------------------------------------
# 6. SPELL CHECK A SENTENCE
# ---------------------------------------------------------

def spell_check(sentence,
                vocabulary,
                unigram_counts,
                bigram_counts):

    words = tokenize(sentence)

    corrected_words = words.copy()

    errors = []

    for i, word in enumerate(words):

        # Word exists in vocabulary
        if word in vocabulary:
            continue

        # Find edit-distance-1 candidates
        candidates = edit_distance_one(
            word,
            vocabulary
        )

        if not candidates:
            errors.append(
                (word, [], None)
            )
            continue

        previous_word = (
            words[i - 1]
            if i > 0
            else None
        )

        next_word = (
            words[i + 1]
            if i < len(words) - 1
            else None
        )

        # Calculate candidate scores
        candidate_scores = {}

        for candidate in candidates:

            score = candidate_score(
                previous_word,
                candidate,
                next_word,
                unigram_counts,
                bigram_counts
            )

            candidate_scores[candidate] = score

        # Best candidate
        best_candidate = max(
            candidate_scores,
            key=candidate_scores.get
        )

        corrected_words[i] = best_candidate

        errors.append(
            (
                word,
                candidate_scores,
                best_candidate
            )
        )

    return corrected_words, errors


# ---------------------------------------------------------
# 7. DISPLAY LANGUAGE MODEL
# ---------------------------------------------------------

def display_model(vocabulary,
                  unigram_counts,
                  bigram_counts):

    print("\nVOCABULARY")
    print("-" * 40)

    print("Number of unique words:",
          len(vocabulary))

    print("\nBIGRAM FREQUENCY TABLE")
    print("-" * 40)

    for (word1, word2), count in bigram_counts.items():
        print(
            f"({word1}, {word2}) : {count}"
        )


# ---------------------------------------------------------
# 8. MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("STATISTICAL SPELL CHECKER")
    print("BIGRAM LANGUAGE MODEL")
    print("=" * 60)

    # -----------------------------------------------------
    # Small corpus
    # Replace this with your actual corpus if required.
    # -----------------------------------------------------

with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read()

    # Build model
    vocabulary, unigram_counts, bigram_counts = \
        build_language_model(corpus)

    print("\nCorpus loaded successfully.")

    print("Vocabulary size:",
          len(vocabulary))

    # Display model
    display_model(
        vocabulary,
        unigram_counts,
        bigram_counts
    )

    # -----------------------------------------------------
    # Input sentence
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    sentence = input(
        "\nEnter a sentence containing possible spelling errors:\n"
    )

    corrected_words, errors = spell_check(
        sentence,
        vocabulary,
        unigram_counts,
        bigram_counts
    )

    # -----------------------------------------------------
    # Display detected errors
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SPELL CHECK RESULTS")
    print("=" * 60)

    found_error = False

    for word, candidates, best in errors:

        if word in vocabulary:
            continue

        found_error = True

        print(f"\nMisspelled word: {word}")

        if not candidates:
            print("Candidates: None")
            print("No correction found.")

        else:
            print("Candidates:")

            for candidate, score in sorted(
                candidates.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(
                    f"  {candidate:<15} "
                    f"Score = {score:.8f}"
                )

            print(
                f"Best candidate: {best}"
            )

    if not found_error:
        print("\nNo spelling errors detected.")

    # -----------------------------------------------------
    # Corrected sentence
    # -----------------------------------------------------

    corrected_sentence = " ".join(corrected_words)

    print("\n" + "=" * 60)
    print("CORRECTED SENTENCE")
    print("=" * 60)

    print(corrected_sentence)


if __name__ == "__main__":
    main()
