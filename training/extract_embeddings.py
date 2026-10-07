import os
import gc
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# FSSM-DDI - MEMORY SAFE PUBMEDBERT EMBEDDING EXTRACTION
# ============================================================

MODEL_NAME = (
    "microsoft/"
    "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)

MAX_LENGTH = 128

# Small batch because WSL has only ~3.7 GB RAM
BATCH_SIZE = 2

# Save after every 500 samples
CHUNK_SIZE = 500


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "preprocessing",
    "processed_train.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "preprocessing",
    "processed_test.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "training",
    "embeddings"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("FSSM-DDI")
print("MEMORY-SAFE PUBMEDBERT EMBEDDING EXTRACTION")
print("=" * 60)

print("\nDevice:", device)
print("Batch size:", BATCH_SIZE)
print("Chunk size:", CHUNK_SIZE)


# ============================================================
# LOAD PUBMEDBERT
# ============================================================

print("\nLoading PubMedBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("PubMedBERT loaded successfully.")


# ============================================================
# EXTRACT DATASET
# ============================================================

def process_dataset(csv_path, dataset_name):

    print("\n" + "=" * 60)
    print("PROCESSING:", dataset_name)
    print("=" * 60)

    df = pd.read_csv(csv_path)

    print("Total samples:", len(df))

    texts = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = (
        df["label"]
        .astype(int)
        .values
    )

    total = len(texts)

    chunk_embeddings = []
    chunk_labels = []

    chunk_number = 0
    chunk_count = 0

    for start in range(0, total, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch_texts = texts[start:end]

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # PubMedBERT inference
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **inputs
            )

            # CLS embedding
            embeddings = (
                outputs.last_hidden_state[:, 0, :]
                .cpu()
            )

        # Store only current chunk
        chunk_embeddings.append(embeddings)

        batch_labels = torch.tensor(
            labels[start:end],
            dtype=torch.long
        )

        chunk_labels.append(
            batch_labels
        )

        chunk_count += len(batch_texts)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"Processed {end}/{total}",
            end="\r"
        )

        # ----------------------------------------------------
        # SAVE CHUNK
        # ----------------------------------------------------

        if (
            chunk_count >= CHUNK_SIZE
            or end == total
        ):

            chunk_number += 1

            embeddings_tensor = torch.cat(
                chunk_embeddings,
                dim=0
            )

            labels_tensor = torch.cat(
                chunk_labels,
                dim=0
            )

            embedding_file = os.path.join(
                OUTPUT_DIR,
                f"{dataset_name}_embeddings_{chunk_number:03d}.pt"
            )

            label_file = os.path.join(
                OUTPUT_DIR,
                f"{dataset_name}_labels_{chunk_number:03d}.pt"
            )

            torch.save(
                embeddings_tensor,
                embedding_file
            )

            torch.save(
                labels_tensor,
                label_file
            )

            print(
                f"\nSaved chunk {chunk_number}: "
                f"{embeddings_tensor.shape}"
            )

            # Clear memory
            del embeddings_tensor
            del labels_tensor

            chunk_embeddings = []
            chunk_labels = []

            chunk_count = 0

            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(
        f"\n{dataset_name} extraction completed."
    )


# ============================================================
# TRAINING DATA
# ============================================================

process_dataset(
    TRAIN_CSV,
    "train"
)


# ============================================================
# TEST DATA
# ============================================================

process_dataset(
    TEST_CSV,
    "test"
)


# ============================================================
# SAVE METADATA
# ============================================================

print("\nSaving metadata...")

metadata_columns = [
    "text",
    "drug1",
    "drug2",
    "ddi",
    "type",
    "label"
]

for dataset_name, csv_path in [
    ("train", TRAIN_CSV),
    ("test", TEST_CSV)
]:

    df = pd.read_csv(csv_path)

    available_columns = [
        column
        for column in metadata_columns
        if column in df.columns
    ]

    metadata_file = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_metadata.csv"
    )

    df[available_columns].to_csv(
        metadata_file,
        index=False
    )

    print(
        "Saved:",
        metadata_file
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING EXTRACTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nYou can now proceed to FSSM model training.")