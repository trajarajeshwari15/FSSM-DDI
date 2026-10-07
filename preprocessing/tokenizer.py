import pandas as pd
import re
import json
import os
from collections import Counter


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "preprocessing",
    "processed_train.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "preprocessing",
    "processed_test.csv"
)

VOCAB_PATH = os.path.join(
    BASE_DIR,
    "preprocessing",
    "vocab.json"
)


# ==========================================
# TOKENIZER
# ==========================================

def tokenize(text):

    text = str(text).lower()

    # Split words and numbers
    tokens = re.findall(
        r"[a-z0-9]+",
        text
    )

    return tokens


# ==========================================
# BUILD VOCABULARY
# ==========================================

def build_vocabulary(train_df):

    counter = Counter()

    for text in train_df["clean_text"]:
        tokens = tokenize(text)
        counter.update(tokens)

    # Special tokens
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    # Add words
    for word, frequency in counter.most_common():

        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


# ==========================================
# CONVERT TEXT TO TOKEN IDs
# ==========================================

def text_to_ids(text, vocab, max_length=128):

    tokens = tokenize(text)

    ids = []

    for token in tokens:

        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab["<UNK>"])

    # Truncate
    ids = ids[:max_length]

    # Padding
    while len(ids) < max_length:
        ids.append(vocab["<PAD>"])

    return ids


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("FSSM - VOCABULARY & TOKENIZATION")
    print("=" * 55)

    # Load processed datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("\nTraining samples:", len(train_df))
    print("Testing samples :", len(test_df))

    # Build vocabulary ONLY from training data
    vocab = build_vocabulary(train_df)

    print("\nVocabulary size:", len(vocab))

    # Convert train text
    train_df["input_ids"] = train_df["clean_text"].apply(
        lambda text: text_to_ids(
            text,
            vocab
        )
    )

    # Convert test text
    test_df["input_ids"] = test_df["clean_text"].apply(
        lambda text: text_to_ids(
            text,
            vocab
        )
    )

    # Save vocabulary
    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vocab,
            file,
            indent=2
        )

    print("\nVocabulary saved to:")
    print(VOCAB_PATH)

    # Show example
    print("\nExample:")
    print("Original:")
    print(train_df["clean_text"].iloc[0])

    print("\nToken IDs:")
    print(train_df["input_ids"].iloc[0])

    print("\nSequence length:")
    print(len(train_df["input_ids"].iloc[0]))

    print("\nTokenization completed successfully.")