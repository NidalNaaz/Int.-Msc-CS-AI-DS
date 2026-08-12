import re
from collections import Counter


# ============================================================
# 1. REGULAR EXPRESSIONS
# ============================================================

def regex_tasks(filename):
    print("\n" + "=" * 60)
    print("1. REGULAR EXPRESSION TASKS")
    print("=" * 60)

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # --------------------------------------------------------
    # 1(a) Two consecutive repeated words
    # --------------------------------------------------------
    pattern_a = r"\b([A-Za-z]+)\s+\1\b"

    print("\n1(a) Consecutive repeated words:")
    matches = re.findall(pattern_a, text, re.IGNORECASE)

    if matches:
        for word in matches:
            print("  ", word, word)
    else:
        print("   No repeated consecutive words found.")

    # --------------------------------------------------------
    # 1(b) Starts with integer and ends with a word
    # --------------------------------------------------------
    pattern_b = r"^\s*\d+.*\b[A-Za-z]+\s*$"

    print("\n1(b) Lines starting with integer and ending with word:")

    for line in text.splitlines():
        if re.fullmatch(pattern_b, line):
            print("  ", line)

    # --------------------------------------------------------
    # 1(c) Contains the words "grotto" and "raven"
    # --------------------------------------------------------
    pattern_c = (
        r"(?i)(?=.*\bgrotto\b)(?=.*\braven\b).*"
    )

    print("\n1(c) Lines containing both 'grotto' and 'raven':")

    for line in text.splitlines():
        if re.search(pattern_c, line):
            print("  ", line)

    # --------------------------------------------------------
    # 1(d) First word of an English sentence
    # --------------------------------------------------------
    # Sentence starts after:
    # beginning of text OR . ! ?
    #
    # Optional quotation/opening punctuation is skipped.
    pattern_d = r"(?:^|[.!?]\s+)[\"'“‘(\[]*([A-Za-z]+)"

    print("\n1(d) First word of each sentence:")

    for match in re.finditer(pattern_d, text):
        first_word = match.group(1)
        print("  ", first_word)


# ============================================================
# 2. ELIZA-LIKE PROGRAM
# ============================================================

def eliza():
    print("\n" + "=" * 60)
    print("2. ELIZA-LIKE PROGRAM")
    print("=" * 60)

    print("ELIZA: Hello! I am a simple study assistant.")
    print("ELIZA: Tell me about your studies.")
    print("ELIZA: Type 'BYE BYE' to exit.\n")

    substitutions = {
        "i am": "you are",
        "i'm": "you are",
        "i was": "you were",
        "i": "you",
        "my": "your",
        "me": "you",
        "you are": "I am",
        "you": "I",
        "your": "my"
    }

    responses = [
        "Why do you say that?",
        "Tell me more about that.",
        "How does that make you feel?",
        "Why do you think that?",
        "Can you explain that further?",
        "That sounds interesting.",
        "What do you think about it?"
    ]

    response_index = 0

    while True:
        user = input("YOU: ").strip()

        if user.upper() == "BYE BYE":
            print("ELIZA: Goodbye!")
            break

        text = user.lower()

        # Rule 1
        if "i feel" in text:
            remainder = text.split("i feel", 1)[1].strip()
            print("ELIZA: Why do you feel " + remainder + "?")
            continue

        # Rule 2
        if "i want" in text:
            remainder = text.split("i want", 1)[1].strip()
            print("ELIZA: Why do you want " + remainder + "?")
            continue

        # Rule 3
        if "because" in text:
            print("ELIZA: Is that the main reason?")
            continue

        # Substitution rule
        transformed = text

        for old, new in substitutions.items():
            transformed = re.sub(
                r"\b" + re.escape(old) + r"\b",
                new,
                transformed
            )

        # Generic response
        print("ELIZA:", responses[response_index])
        response_index = (response_index + 1) % len(responses)


# ============================================================
# 3. RULE-BASED ENGLISH TOKENIZER
# ============================================================

def tokenize(text):
    """
    Tokenization rules:

    1. Abbreviations:
       U.S.A.
       U.K.
       e.g.
       Dr.

    2. Internal hyphenation:
       ice-cream
       state-of-the-art

    3. Contractions:
       isn't -> is + n't
       can't -> ca + n't
       I'm -> I + 'm
       we're -> we + 're

    4. Words
    5. Numbers
    6. Punctuation and symbols
    """

    # Combined regular expression.
    pattern = r"""
        # Abbreviations such as U.S.A. or U.S.
        \b(?:[A-Za-z]\.){2,}

        |

        # Common abbreviations such as e.g. or Dr.
        \b(?:[A-Za-z]{1,4}\.){1,}

        |

        # Hyphenated words
        \b[A-Za-z]+(?:-[A-Za-z]+)+\b

        |

        # Contractions with n't
        \b[A-Za-z]+n't\b

        |

        # Other contractions
        \b[A-Za-z]+(?:'re|'ve|'ll|'d|'m|'s)\b

        |

        # Normal words
        \b[A-Za-z]+\b

        |

        # Numbers
        \b\d+(?:\.\d+)?\b

        |

        # Individual punctuation / symbols
        [^\w\s]
    """

    raw_tokens = re.findall(pattern, text, re.VERBOSE)

    final_tokens = []

    for token in raw_tokens:

        # ----------------------------------------------------
        # Split contractions
        # ----------------------------------------------------
        match = re.fullmatch(r"([A-Za-z]+)(n't)", token)

        if match:
            final_tokens.extend([
                match.group(1),
                match.group(2)
            ])
            continue

        match = re.fullmatch(
            r"([A-Za-z]+)('re|'ve|'ll|'d|'m|'s)",
            token
        )

        if match:
            final_tokens.extend([
                match.group(1),
                match.group(2)
            ])
            continue

        final_tokens.append(token)

    return final_tokens


def tokenizer_program():
    print("\n" + "=" * 60)
    print("3. ENGLISH TOKENIZER")
    print("=" * 60)

    text = input("\nEnter text:\n")

    tokens = tokenize(text)

    print("\nTokens:")
    for i, token in enumerate(tokens, 1):
        print(f"{i:2}. {token}")


# ============================================================
# 4. FINITE STATE AUTOMATON
# ============================================================

def plural_fsa(word):
    """
    Accepts valid English plural forms derived from singular
    nouns ending in 'y'.

    Examples:
        boy     -> boys
        toy     -> toys
        pony    -> ponies
        sky     -> skies
        puppy   -> puppies

    Invalid:
        boies
        toies
        ponys
    """

    word = word.lower()

    # Must end in s
    if not word.endswith("s"):
        return False

    singular = word[:-1]

    # Singular must end in y
    if not singular.endswith("y"):
        return False

    if len(singular) < 2:
        return False

    before_y = singular[-2]

    vowels = "aeiou"

    # FSA branch:
    #
    # vowel + y + s
    #       -> boys, toys
    #
    # consonant + y
    #       -> ies
    #       -> ponies, puppies, skies

    if before_y in vowels:

        # Correct form must be ys
        return word.endswith("ys")

    else:

        # Correct form must be ies
        return word.endswith("ies")


def show_fsa():
    print("\n" + "=" * 60)
    print("4. FINITE STATE AUTOMATON")
    print("=" * 60)

    test_words = [
        "boys",
        "toys",
        "ponies",
        "skies",
        "puppies",
        "boies",
        "toies",
        "ponys",
        "dog",
        "cats"
    ]

    for word in test_words:
        result = "ACCEPT" if plural_fsa(word) else "REJECT"
        print(f"{word:10} -> {result}")

    print("\nEnter your own words (type STOP to finish):")

    while True:
        word = input("> ").strip()

        if word.upper() == "STOP":
            break

        result = "ACCEPT" if plural_fsa(word) else "REJECT"
        print(result)


# ============================================================
# 5. FINITE STATE TRANSDUCER
# ============================================================

def plural_fst(lexical_form):
    """
    Implements:

        ε => e / {x,s,z}^ __ s#

    Meaning:

        If the stem ends in x, s or z and the lexical
        suffix is ^s#, insert 'e' before s.

    Examples:

        fox^s# -> foxes
        boy^s# -> boys
        bus^s# -> buses
        quiz^s# -> quizzes
    """

    # Validate lexical form
    match = re.fullmatch(r"(.+)\^s#", lexical_form.lower())

    if not match:
        return None

    stem = match.group(1)

    # e-insertion rule
    if stem.endswith(("x", "s", "z")):
        return stem + "es"

    return stem + "s"


def show_fst():
    print("\n" + "=" * 60)
    print("5. FINITE STATE TRANSDUCER")
    print("=" * 60)

    inputs = [
        "fox^s#",
        "boy^s#",
        "bus^s#",
        "quiz^s#",
        "cat^s#",
        "box^s#"
    ]

    print(f"{'INPUT':15} {'OUTPUT'}")
    print("-" * 30)

    for lexical in inputs:
        output = plural_fst(lexical)
        print(f"{lexical:15} {output}")

    print("\nEnter lexical forms manually.")
    print("Example: fox^s#")
    print("Type STOP to finish.")

    while True:
        lexical = input("> ").strip()

        if lexical.upper() == "STOP":
            break

        output = plural_fst(lexical)

        if output is None:
            print("Invalid lexical form.")
        else:
            print("Output:", output)


# ============================================================
# 6. BYTE PAIR ENCODING TOKENIZER
# ============================================================

def get_vocab(corpus):
    """
    Convert words into character sequences with </w>
    marking the end of a word.
    """

    words = re.findall(r"[A-Za-z]+", corpus.lower())

    vocab = Counter()

    for word in words:
        symbols = tuple(list(word) + ["</w>"])
        vocab[symbols] += 1

    return vocab


def get_pair_statistics(vocab):
    pairs = Counter()

    for symbols, frequency in vocab.items():

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += frequency

    return pairs


def merge_pair(pair, vocab):
    """
    Merge the most frequent pair.
    """

    new_vocab = Counter()

    first, second = pair

    for symbols, frequency in vocab.items():

        new_symbols = []
        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and symbols[i] == first
                and symbols[i + 1] == second
            ):
                new_symbols.append(first + second)
                i += 2

            else:
                new_symbols.append(symbols[i])
                i += 1

        new_vocab[tuple(new_symbols)] += frequency

    return new_vocab


def print_vocab(vocab):
    for symbols, frequency in vocab.items():
        print(f"  {' '.join(symbols):30} {frequency}")


def bpe_program():
    print("\n" + "=" * 60)
    print("6. BYTE PAIR ENCODING")
    print("=" * 60)

    corpus = """
    low lower lowest
    low lower lower
    new newer newest
    """

    print("\nRepresentative corpus:")
    print(corpus.strip())

    vocab = get_vocab(corpus)

    print("\nInitial vocabulary:")
    print_vocab(vocab)

    num_merges = 8

    for step in range(1, num_merges + 1):

        pairs = get_pair_statistics(vocab)

        if not pairs:
            break

        best_pair, frequency = pairs.most_common(1)[0]

        print("\n" + "-" * 50)
        print(f"Step {step}")
        print("-" * 50)

        print(
            f"Most frequent pair: "
            f"('{best_pair[0]}', '{best_pair[1]}')"
        )

        print(f"Frequency: {frequency}")

        vocab = merge_pair(best_pair, vocab)

        print("\nVocabulary after merge:")
        print_vocab(vocab)

    print("\nFinal BPE vocabulary:")
    print_vocab(vocab)

    print("\nBPE merge process completed.")


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print("\n")
        print("=" * 60)
        print("NLP LAB PROGRAM")
        print("=" * 60)

        print("1. Regular Expressions")
        print("2. ELIZA-like Program")
        print("3. Rule-based Tokenizer")
        print("4. Finite State Automaton")
        print("5. Finite State Transducer")
        print("6. Byte Pair Encoding")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            filename = input(
                "Enter input filename "
                "(default: input.txt): "
            ).strip()

            if not filename:
                filename = "input.txt"

            try:
                regex_tasks(filename)
            except FileNotFoundError:
                print(f"File '{filename}' not found.")

        elif choice == "2":
            eliza()

        elif choice == "3":
            tokenizer_program()

        elif choice == "4":
            show_fsa()

        elif choice == "5":
            show_fst()

        elif choice == "6":
            bpe_program()

        elif choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
