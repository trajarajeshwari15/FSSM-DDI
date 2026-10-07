import os
import sys
import glob
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT MODEL
# ============================================================

from model.fssm_model import FSSMModel


# ============================================================
# PATHS
# ============================================================

EMBEDDING_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "embeddings"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "model"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 0.0005

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS WEIGHTS
#
# Training distribution:
#
# Class 0 = 11985
# Class 1 =   379
# Class 2 =   860
# Class 3 =   115
# Class 4 =   776
# ============================================================

class_counts = torch.tensor(
    [
        11985,
        379,
        860,
        115,
        776
    ],
    dtype=torch.float
)


# Balanced inverse-frequency weights
class_weights = (
    class_counts.sum()
    /
    (
        len(class_counts)
        * class_counts
    )
)


class_weights = class_weights.to(
    DEVICE
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("FSSM-DDI CLASS-BALANCED TRAINING")
print("=" * 60)

print()
print("Device        :", DEVICE)
print("Batch size    :", BATCH_SIZE)
print("Epochs        :", EPOCHS)
print("Learning rate :", LEARNING_RATE)

print()
print("Class counts:")
print(class_counts.tolist())

print()
print("Class weights:")
print(class_weights.tolist())


# ============================================================
# FIND TRAINING CHUNKS
# ============================================================

train_embedding_files = sorted(
    glob.glob(
        os.path.join(
            EMBEDDING_DIR,
            "train_embeddings_*.pt"
        )
    )
)

train_label_files = sorted(
    glob.glob(
        os.path.join(
            EMBEDDING_DIR,
            "train_labels_*.pt"
        )
    )
)


print()
print(
    "Training embedding chunks:",
    len(train_embedding_files)
)

print(
    "Training label chunks:",
    len(train_label_files)
)


# ============================================================
# CHECK
# ============================================================

if len(train_embedding_files) == 0:
    raise FileNotFoundError(
        "Training embeddings not found."
    )


if len(train_embedding_files) != len(
    train_label_files
):
    raise RuntimeError(
        "Embedding and label chunk counts "
        "do not match."
    )


# ============================================================
# MODEL
# ============================================================

print()
print("Creating FSSM model...")

model = FSSMModel(
    input_dim=768,
    hidden_dim=300,
    state_size=16,
    num_classes=5
)

model = model.to(DEVICE)

print(
    "Model created successfully."
)


# ============================================================
# WEIGHTED LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

for epoch in range(EPOCHS):

    print()
    print("=" * 60)
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 60)

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    for chunk_index in range(
        len(train_embedding_files)
    ):

        embedding_file = (
            train_embedding_files[
                chunk_index
            ]
        )

        label_file = (
            train_label_files[
                chunk_index
            ]
        )

        print()
        print(
            f"Loading chunk "
            f"{chunk_index + 1}/"
            f"{len(train_embedding_files)}"
        )


        # ----------------------------------------------------
        # LOAD CHUNK
        # ----------------------------------------------------

        embeddings = torch.load(
            embedding_file,
            map_location="cpu"
        )

        labels = torch.load(
            label_file,
            map_location="cpu"
        )


        print(
            "Embedding shape:",
            tuple(embeddings.shape)
        )

        print(
            "Label shape:",
            tuple(labels.shape)
        )


        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        dataset = TensorDataset(
            embeddings,
            labels
        )


        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )


        # ----------------------------------------------------
        # BATCH TRAINING
        # ----------------------------------------------------

        for batch_number, (
            batch_embeddings,
            batch_labels
        ) in enumerate(loader):

            batch_embeddings = (
                batch_embeddings.to(DEVICE)
            )

            batch_labels = (
                batch_labels.to(DEVICE)
            )


            optimizer.zero_grad()


            outputs = model(
                batch_embeddings
            )


            loss = criterion(
                outputs,
                batch_labels
            )


            loss.backward()

            optimizer.step()


            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            batch_size_now = (
                batch_labels.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size_now
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            total_correct += (
                predictions == batch_labels
            ).sum().item()


            total_samples += (
                batch_size_now
            )


            if (
                batch_number + 1
            ) % 10 == 0:

                print(
                    f"  Batch "
                    f"{batch_number + 1}/"
                    f"{len(loader)}"
                    f" | Loss: "
                    f"{loss.item():.4f}"
                )


        # ----------------------------------------------------
        # MEMORY CLEANUP
        # ----------------------------------------------------

        del embeddings
        del labels
        del dataset
        del loader

        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


        print(
            f"Completed chunk "
            f"{chunk_index + 1}/"
            f"{len(train_embedding_files)}"
        )


    # ========================================================
    # EPOCH RESULTS
    # ========================================================

    epoch_loss = (
        total_loss /
        total_samples
    )

    epoch_accuracy = (
        total_correct /
        total_samples
    ) * 100


    print()
    print("-" * 60)

    print(
        f"Epoch {epoch + 1} Result"
    )

    print("-" * 60)

    print(
        f"Loss     : {epoch_loss:.4f}"
    )

    print(
        f"Accuracy : {epoch_accuracy:.2f}%"
    )

    print(
        f"Samples  : {total_samples}"
    )


# ============================================================
# SAVE BALANCED MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fssm_ddi_model_balanced.pt"
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("CLASS-BALANCED TRAINING COMPLETED")
print("=" * 60)

print()
print("Balanced model saved at:")

print(
    MODEL_PATH
)