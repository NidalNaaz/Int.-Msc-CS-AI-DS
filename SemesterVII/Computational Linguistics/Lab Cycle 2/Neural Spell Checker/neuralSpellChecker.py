import re
import random
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# SETTINGS
# =========================================================

random.seed(42)
torch.manual_seed(42)

EPOCHS = 30
LEARNING_RATE = 0.003
EMBED_SIZE = 32
HIDDEN_SIZE = 64


# =========================================================
# 1. LOAD CORPUS
# =========================================================

def load_words():

    try:
        with open(
            "corpus.txt",
            "r",
            encoding="utf-8"
        ) as file:
            text = file.read()

    except FileNotFoundError:
        print("ERROR: corpus.txt not found.")
        return []

    words = re.findall(
        r"[a-z]+",
        text.lower()
    )

    return sorted(set(words))


# =========================================================
# 2. CREATE ARTIFICIAL SPELLING ERROR
# =========================================================

def corrupt_word(word):

    if len(word) < 3:
        return word

    operation = random.choice([
        "delete",
        "insert",
        "substitute",
        "transpose"
    ])

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    # -----------------------------
    # DELETE
    # -----------------------------

    if operation == "delete":

        i = random.randrange(len(word))

        return (
            word[:i] +
            word[i + 1:]
        )

    # -----------------------------
    # INSERT
    # -----------------------------

    if operation == "insert":

        i = random.randrange(len(word) + 1)

        char = random.choice(alphabet)

        return (
            word[:i] +
            char +
            word[i:]
        )

    # -----------------------------
    # SUBSTITUTE
    # -----------------------------

    if operation == "substitute":

        i = random.randrange(len(word))

        char = random.choice(alphabet)

        return (
            word[:i] +
            char +
            word[i + 1:]
        )

    # -----------------------------
    # TRANSPOSE
    # -----------------------------

    if operation == "transpose":

        i = random.randrange(len(word) - 1)

        chars = list(word)

        chars[i], chars[i + 1] = \
            chars[i + 1], chars[i]

        return "".join(chars)

    return word


# =========================================================
# 3. CHARACTER VOCABULARY
# =========================================================

SPECIAL = [
    "<PAD>",
    "<SOS>",
    "<EOS>"
]


def create_vocabulary(words):

    characters = set(
        "".join(words)
    )

    chars = SPECIAL + sorted(
        characters
    )

    char_to_id = {
        char: i
        for i, char in enumerate(chars)
    }

    id_to_char = {
        i: char
        for char, i in char_to_id.items()
    }

    return char_to_id, id_to_char


# =========================================================
# 4. ENCODE WORD
# =========================================================

def encode_word(
        word,
        char_to_id):

    return [
        char_to_id["<SOS>"]
    ] + [
        char_to_id[c]
        for c in word
    ] + [
        char_to_id["<EOS>"]
    ]


# =========================================================
# 5. CREATE DATASET
# =========================================================

def create_dataset(
        words,
        char_to_id):

    dataset = []

    for word in words:

        # Generate multiple corrupted versions
        for _ in range(5):

            corrupted = corrupt_word(
                word
            )

            source = encode_word(
                corrupted,
                char_to_id
            )

            target = encode_word(
                word,
                char_to_id
            )

            dataset.append(
                (source, target)
            )

    return dataset


# =========================================================
# 6. ENCODER
# =========================================================

class Encoder(nn.Module):

    def __init__(
            self,
            vocab_size,
            embed_size,
            hidden_size):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size
        )

    def forward(self, x):

        embedded = self.embedding(x)

        outputs, (hidden, cell) = \
            self.lstm(embedded)

        return hidden, cell


# =========================================================
# 7. DECODER
# =========================================================

class Decoder(nn.Module):

    def __init__(
            self,
            vocab_size,
            embed_size,
            hidden_size):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(
            self,
            x,
            hidden,
            cell):

        x = x.unsqueeze(0)

        embedded = self.embedding(x)

        output, (hidden, cell) = \
            self.lstm(
                embedded,
                (hidden, cell)
            )

        prediction = self.fc(
            output.squeeze(0)
        )

        return prediction, hidden, cell


# =========================================================
# 8. SEQ2SEQ MODEL
# =========================================================

class Seq2Seq(nn.Module):

    def __init__(
            self,
            encoder,
            decoder):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            source,
            target,
            teacher_forcing_ratio=0.5):

        target_length = len(target)

        vocab_size = (
            self.decoder.fc.out_features
        )

        outputs = torch.zeros(
            target_length,
            vocab_size
        )

        hidden, cell = self.encoder(
            source
        )

        input_char = target[0]

        for t in range(
                1,
                target_length):

            output, hidden, cell = \
                self.decoder(
                    input_char,
                    hidden,
                    cell
                )

            outputs[t] = output

            best_guess = output.argmax(0)

            if random.random() < \
                    teacher_forcing_ratio:

                input_char = target[t]

            else:

                input_char = best_guess

        return outputs


# =========================================================
# 9. TRAIN MODEL
# =========================================================

def train_model(
        model,
        dataset,
        criterion,
        optimizer):

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        random.shuffle(dataset)

        for source, target in dataset:

            source = torch.tensor(
                source,
                dtype=torch.long
            )

            target = torch.tensor(
                target,
                dtype=torch.long
            )

            optimizer.zero_grad()

            output = model(
                source,
                target
            )

            loss = criterion(
                output[1:].reshape(
                    -1,
                    output.shape[-1]
                ),
                target[1:]
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataset)
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss: {average_loss:.4f}"
        )


# =========================================================
# 10. CORRECT WORD
# =========================================================

def correct_word(
        word,
        model,
        char_to_id,
        id_to_char,
        max_length=20):

    model.eval()

    source = encode_word(
        word,
        char_to_id
    )

    source = torch.tensor(
        source,
        dtype=torch.long
    )

    with torch.no_grad():

        hidden, cell = model.encoder(
            source
        )

    input_char = torch.tensor(
        char_to_id["<SOS>"],
        dtype=torch.long
    )

    result = []

    for _ in range(max_length):

        with torch.no_grad():

            output, hidden, cell = \
                model.decoder(
                    input_char,
                    hidden,
                    cell
                )

        prediction = output.argmax(
            1
        ).item()

        character = id_to_char[
            prediction
        ]

        if character == "<EOS>":
            break

        if character not in SPECIAL:
            result.append(character)

        input_char = torch.tensor(
            prediction,
            dtype=torch.long
        )

    return "".join(result)


# =========================================================
# 11. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("NEURAL SPELLING ERROR CORRECTOR")
    print("CHARACTER-LEVEL LSTM - PYTORCH")
    print("=" * 60)

    # -----------------------------------------------------
    # Load corpus
    # -----------------------------------------------------

    words = load_words()

    if not words:
        return

    print(
        "\nVocabulary:",
        len(words),
        "words"
    )

    # -----------------------------------------------------
    # Character vocabulary
    # -----------------------------------------------------

    char_to_id, id_to_char = \
        create_vocabulary(words)

    vocab_size = len(char_to_id)

    print(
        "Character vocabulary:",
        vocab_size
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = create_dataset(
        words,
        char_to_id
    )

    print(
        "Training examples:",
        len(dataset)
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    encoder = Encoder(
        vocab_size,
        EMBED_SIZE,
        HIDDEN_SIZE
    )

    decoder = Decoder(
        vocab_size,
        EMBED_SIZE,
        HIDDEN_SIZE
    )

    model = Seq2Seq(
        encoder,
        decoder
    )

    # -----------------------------------------------------
    # Loss + optimizer
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=char_to_id["<PAD>"]
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print("\nTraining neural model...\n")

    train_model(
        model,
        dataset,
        criterion,
        optimizer
    )

    # -----------------------------------------------------
    # Interactive testing
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("NEURAL SPELL CHECK")
    print("=" * 60)

    while True:

        word = input(
            "\nEnter a misspelled word "
            "(or 'quit'):\n"
        ).lower()

        if word == "quit":
            break

        if not word.isalpha():
            print(
                "Please enter alphabetic words."
            )
            continue

        correction = correct_word(
            word,
            model,
            char_to_id,
            id_to_char
        )

        print(
            "Neural correction:",
            correction
        )


if __name__ == "__main__":
    main()
