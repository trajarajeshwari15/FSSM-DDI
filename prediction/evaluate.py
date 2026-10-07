import os
import sys
import glob
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "fssm_ddi_model_balanced.pt"
)

RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "prediction",
    "balanced_evaluation_results.txt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("FSSM-DDI BALANCED MODEL EVALUATION")
print("=" * 60)

print()
print("Device:", DEVICE)

print()
print("Loading balanced model...")


# ============================================================
# MODEL
# ============================================================

model = FSSMModel(
    input_dim=768,
    hidden_dim=300,
    state_size=16,
    num_classes=5
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print(
    "Balanced model loaded successfully."
)


# ============================================================
# FIND TEST CHUNKS
# ============================================================

test_embedding_files = sorted(
    glob.glob(
        os.path.join(
            EMBEDDING_DIR,
            "test_embeddings_*.pt"
        )
    )
)

test_label_files = sorted(
    glob.glob(
        os.path.join(
            EMBEDDING_DIR,
            "test_labels_*.pt"
        )
    )
)


print()
print(
    "Test embedding chunks:",
    len(test_embedding_files)
)

print(
    "Test label chunks:",
    len(test_label_files)
)


# ============================================================
# STORAGE
# ============================================================

all_predictions = []

all_labels = []


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 60)
print("STARTING TEST EVALUATION")
print("=" * 60)


with torch.no_grad():

    for i in range(
        len(test_embedding_files)
    ):

        print()
        print(
            f"Loading chunk "
            f"{i + 1}/"
            f"{len(test_embedding_files)}"
        )


        embeddings = torch.load(
            test_embedding_files[i],
            map_location="cpu"
        )

        labels = torch.load(
            test_label_files[i],
            map_location="cpu"
        )


        embeddings = embeddings.to(
            DEVICE
        )


        outputs = model(
            embeddings
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.tolist()
        )


        print(
            "Processed:",
            len(labels),
            "samples"
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4],
    target_names=[
        "No Interaction",
        "Interaction Type 1",
        "Interaction Type 2",
        "Interaction Type 3",
        "Interaction Type 4"
    ],
    zero_division=0
)


matrix = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("BALANCED MODEL RESULTS")
print("=" * 60)

print()

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    report
)


print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    matrix
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULT_PATH,
    "w"
) as f:

    f.write(
        "FSSM-DDI BALANCED MODEL EVALUATION\n"
    )

    f.write(
        "=" * 60
    )

    f.write("\n\n")

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write("\n")

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 60
    )

    f.write("\n\n")

    f.write(
        report
    )

    f.write("\n\n")

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "=" * 60
    )

    f.write("\n\n")

    f.write(
        str(matrix)
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print(
    "Results saved at:"
)

print(
    RESULT_PATH
)