import pandas as pd
import re
import os

# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR, "dataset", "train_smiles_pos.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR, "dataset", "test_smiles_pos.csv"
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================
# TEXT PREPROCESSING
# ==============================

def clean_text(text, drug1, drug2):

    if pd.isna(text):
        text = ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace drug names with standard markers
    if pd.notna(drug1):
        drug1 = str(drug1).lower().strip()
        if drug1:
            text = re.sub(
                r"\b" + re.escape(drug1) + r"\b",
                "drug1",
                text
            )

    if pd.notna(drug2):
        drug2 = str(drug2).lower().strip()
        if drug2:
            text = re.sub(
                r"\b" + re.escape(drug2) + r"\b",
                "drug2",
                text
            )

    # Keep words, numbers and important drug markers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==============================
# LOAD DATASET
# ==============================

def load_dataset(path):

    print(f"\nLoading: {path}")

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    required_columns = [
        "text",
        "drug1",
        "drug2",
        "ddi",
        "type",
        "label",
        "smile1",
        "smile2",
        "pos1",
        "pos2"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ==============================
# PROCESS DATASET
# ==============================

def process_dataset(df):

    # Create cleaned text
    df["clean_text"] = df.apply(
        lambda row: clean_text(
            row["text"],
            row["drug1"],
            row["drug2"]
        ),
        axis=1
    )

    # Convert label to integer
    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    # Remove rows with missing labels
    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Keep only valid labels
    df = df[df["label"].isin([0, 1, 2, 3, 4])]

    return df


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print("=" * 50)
    print("FSSM - DDI DATA PREPROCESSING")
    print("=" * 50)

    # Load train and test
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    # Process
    train_df = process_dataset(train_df)
    test_df = process_dataset(test_df)

    # Save processed files
    train_output = os.path.join(
        OUTPUT_DIR,
        "processed_train.csv"
    )

    test_output = os.path.join(
        OUTPUT_DIR,
        "processed_test.csv"
    )

    train_df.to_csv(
        train_output,
        index=False
    )

    test_df.to_csv(
        test_output,
        index=False
    )

    # ==============================
    # DISPLAY INFORMATION
    # ==============================

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED")
    print("=" * 50)

    print("\nTraining samples:", len(train_df))
    print("Testing samples :", len(test_df))

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nTesting label distribution:")
    print(test_df["label"].value_counts().sort_index())

    print("\nLabel mapping:")
    print("0 -> Negative")
    print("1 -> Effect")
    print("2 -> Advice")
    print("3 -> Interaction")
    print("4 -> Mechanism")

    print("\nSample processed text:")
    print(train_df["clean_text"].iloc[0])

    print("\nSaved:")
    print(train_output)
    print(test_output)