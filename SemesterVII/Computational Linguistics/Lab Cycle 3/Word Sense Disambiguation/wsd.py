import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    """Tokenizes text into lowercase words, stripping punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class NaiveBayesWSD:

    def __init__(self):
        self.classes = []
        self.priors = {}
        self.word_counts = defaultdict(Counter)
        self.total_words_in_class = {}
        self.vocab = set()

    def fit(self, training_data):
        """Trains the Naive Bayes Classifier on labeled text samples."""
        total_docs = len(training_data)
        doc_counts = Counter()

        for text, sense in training_data:
            doc_counts[sense] += 1
            tokens = tokenize(text)
            for word in tokens:
                self.word_counts[sense][word] += 1
                self.vocab.add(word)

        self.classes = list(doc_counts.keys())

        # Prior Probability P(Sense)
        for sense in self.classes:
            self.priors[sense] = doc_counts[sense] / total_docs
            self.total_words_in_class[sense] = sum(
                self.word_counts[sense].values()
            )

    def predict(self, test_sentence):
        """Predicts the sense of a target word in the test sentence using Add-1 smoothing."""
        test_tokens = tokenize(test_sentence)
        vocab_size = len(self.vocab)
        log_probs = {}

        for sense in self.classes:
            # Start with log(P(Sense)) to prevent numerical underflow
            log_prob = math.log(self.priors[sense])

            for word in test_tokens:
                # Count of word in given sense
                count_w_c = self.word_counts[sense][word]

                # P(word|sense) with Add-1 (Laplace) Smoothing
                # (count(w, c) + 1) / (total_words_in_c + |V|)
                p_word_given_sense = (count_w_c + 1) / (
                    self.total_words_in_class[sense] + vocab_size
                )
                log_prob += math.log(p_word_given_sense)

            log_probs[sense] = log_prob

        # Return the class with maximum posterior probability
        best_sense = max(log_probs, key=log_probs.get)
        return best_sense, log_probs


if __name__ == "__main__":
    # Training Data from Table[cite: 2]
    training_data = [
        ("I love fish. The smoked bass fish was delicious.", "fish"),
        ("The bass fish swam along the line.", "fish"),
        ("He hauled in a big catch of smoked bass fish.", "fish"),
        ("The bass guitar player played a smooth jazz line.", "guitar"),
    ]

    # Test Data[cite: 2]
    test_sentence = "He loves jazz. The bass_line provided the foundation for the guitar solo in the jazz piece"
    test_target_word = "bass"

    # Train Classifier
    model = NaiveBayesWSD()
    model.fit(training_data)

    # Predict Sense
    predicted_sense, log_probs = model.predict(test_sentence)

    # Print Results
    print("=" * 65)
    print(" NAIVE BAYES WSD CLASSIFIER (ADD-1 SMOOTHING)")
    print("=" * 65)
    print(f"Target Word   : '{test_target_word}'")
    print(f"Test Sentence : '{test_sentence}'\n")

    print("-" * 65)
    print(" LOG POSTERIOR PROBABILITIES")
    print("-" * 65)
    for sense, log_p in log_probs.items():
        print(f"Sense: {sense:<10} | Log-Likelihood: {log_p:.4f}")

    print("-" * 65)
    print(f"Predicted Sense Output : {predicted_sense}")
    print("=" * 65)
