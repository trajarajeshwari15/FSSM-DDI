import os
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


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
# FSSM MODEL
# ============================================================

from model.fssm_model import FSSMModel


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "test.csv"
)

# IMPORTANT:
# We are using the NEW BALANCED MODEL
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "fssm_ddi_model_balanced.pt"
)


# ============================================================
# PUBMEDBERT
# ============================================================

PUBMEDBERT_NAME = (
    "microsoft/"
    "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "No Interaction",
    1: "Interaction Type 1",
    2: "Interaction Type 2",
    3: "Interaction Type 3",
    4: "Interaction Type 4"
}


# ============================================================
# START
# ============================================================

print("=" * 60)
print("FSSM-DDI DRUG INTERACTION PREDICTION")
print("=" * 60)

print()

print("Device:", DEVICE)


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: Balanced model not found!")
    print()
    print("Expected model:")
    print(MODEL_PATH)
    print()

    print(
        "Please make sure balanced training "
        "was completed successfully."
    )

    sys.exit(1)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset loaded:",
    len(df),
    "records"
)


# ============================================================
# LOAD PUBMEDBERT
# ============================================================

print()
print("Loading PubMedBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    PUBMEDBERT_NAME
)

bert_model = AutoModel.from_pretrained(
    PUBMEDBERT_NAME
)

bert_model = bert_model.to(
    DEVICE
)

bert_model.eval()

print(
    "PubMedBERT loaded successfully."
)


# ============================================================
# LOAD BALANCED FSSM MODEL
# ============================================================

print()
print("Loading BALANCED FSSM model...")

fssm_model = FSSMModel(
    input_dim=768,
    hidden_dim=300,
    state_size=16,
    num_classes=5
)

fssm_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

fssm_model = fssm_model.to(
    DEVICE
)

fssm_model.eval()

print(
    "Balanced FSSM model loaded successfully."
)


# ============================================================
# FIND DRUG PAIR
# ============================================================

def find_drug_pair(
    drug1,
    drug2
):

    drug1 = drug1.lower().strip()
    drug2 = drug2.lower().strip()

    result = df[
        (
            (df["drug1"].astype(str).str.lower() == drug1)
            &
            (df["drug2"].astype(str).str.lower() == drug2)
        )
        |
        (
            (df["drug1"].astype(str).str.lower() == drug2)
            &
            (df["drug2"].astype(str).str.lower() == drug1)
        )
    ]

    return result


# ============================================================
# CREATE PUBMEDBERT EMBEDDING
# ============================================================

def create_embedding(
    text
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = bert_model(
            **inputs
        )

        embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

    return embedding


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_interaction(
    drug1,
    drug2
):

    matches = find_drug_pair(
        drug1,
        drug2
    )


    # --------------------------------------------------------
    # PAIR NOT FOUND
    # --------------------------------------------------------

    if len(matches) == 0:

        print()
        print("-" * 60)
        print("DRUG PAIR NOT FOUND")
        print("-" * 60)

        print()

        print(
            f"{drug1} + {drug2}"
        )

        print()

        print(
            "This drug pair is not available "
            "in the current dataset."
        )

        print(
            "Please enter a drug pair "
            "present in the dataset."
        )

        return None


    # --------------------------------------------------------
    # SELECT FIRST MATCH
    # --------------------------------------------------------

    row = matches.iloc[0]

    text = str(
        row["text"]
    )


    # --------------------------------------------------------
    # INPUT INFORMATION
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("INPUT")
    print("-" * 60)

    print()

    print(
        "Drug 1:",
        row["drug1"]
    )

    print(
        "Drug 2:",
        row["drug2"]
    )

    print()

    print(
        "Dataset text:"
    )

    print(text)


    # --------------------------------------------------------
    # PUBMEDBERT
    # --------------------------------------------------------

    embedding = create_embedding(
        text
    )

    print()

    print(
        "Embedding shape:",
        tuple(embedding.shape)
    )


    # --------------------------------------------------------
    # FSSM
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = fssm_model(
            embedding
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = (
        probabilities[
            0,
            predicted_class
        ].item()
        * 100
    )


    # --------------------------------------------------------
    # ACTUAL DATASET LABEL
    # --------------------------------------------------------

    actual_label = int(
        row["label"]
    )

    actual_type = str(
        row["type"]
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print()

    print(
        "Drug Pair:",
        drug1,
        "+",
        drug2
    )

    print()

    print(
        "Predicted Class:",
        predicted_class
    )

    print(
        "Prediction:",
        CLASS_NAMES[
            predicted_class
        ]
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print()

    print(
        "Actual Dataset Class:",
        actual_label
    )

    print(
        "Actual Dataset Type:",
        actual_type
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    print()
    print("-" * 60)
    print("CLASS PROBABILITIES")
    print("-" * 60)

    for class_id in range(5):

        probability = (
            probabilities[
                0,
                class_id
            ].item()
            * 100
        )

        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]}: "
            f"{probability:.2f}%"
        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    print()
    print("-" * 60)
    print("EXPLANATION")
    print("-" * 60)

    if predicted_class == 0:

        print(
            "The model predicts NO INTERACTION "
            "between the two drugs."
        )

        print(
            "The prediction is based on the "
            "learned representation of the "
            "drug-pair description."
        )

    else:

        print(
            "The model predicts an INTERACTION "
            "between the two drugs."
        )

        print(
            f"Predicted interaction class: "
            f"{predicted_class}"
        )

        print(
            f"Dataset interaction type: "
            f"{actual_type}"
        )

        print(
            "The confidence score represents "
            "the probability assigned to the "
            "predicted class."
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "drug1": drug1,
        "drug2": drug2,
        "predicted_class": predicted_class,
        "prediction": CLASS_NAMES[
            predicted_class
        ],
        "confidence": confidence,
        "actual_label": actual_label,
        "actual_type": actual_type
    }


# ============================================================
# INTERACTIVE MODE
# ============================================================

print()
print("=" * 60)
print("READY FOR DRUG PREDICTION")
print("=" * 60)

print()

print(
    "Enter drug names exactly as they "
    "appear in the dataset."
)

print(
    "Type 'exit' to stop."
)


while True:

    print()

    drug1 = input(
        "Enter Drug 1: "
    ).strip()


    if drug1.lower() == "exit":
        break


    drug2 = input(
        "Enter Drug 2: "
    ).strip()


    if drug2.lower() == "exit":
        break


    predict_interaction(
        drug1,
        drug2
    )


print()

print(
    "Prediction program closed."
)