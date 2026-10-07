import os
import sys

import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from model.fssm_model import FSSMModel

app = Flask(__name__)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "training",
    "embeddings",
    "test_metadata.csv"
)

EMBEDDING_PATH = os.path.join(
    PROJECT_ROOT,
    "training",
    "test_embeddings.pt"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "fssm_ddi_model_balanced.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    0: "No Interaction",
    1: "Interaction Type 1",
    2: "Interaction Type 2",
    3: "Interaction Type 3",
    4: "Interaction Type 4"
}

print("=" * 60)
print("FSSM-DDI WEB APPLICATION")
print("=" * 60)

# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading embedding metadata...")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        f"Embedding metadata not found:\n{METADATA_PATH}"
    )

metadata = pd.read_csv(METADATA_PATH)

print(f"Metadata loaded: {len(metadata)} records")

# ============================================================
# LOAD PRECOMPUTED EMBEDDINGS
# ============================================================

print("\nLoading precomputed PubMedBERT embeddings...")

if not os.path.exists(EMBEDDING_PATH):
    raise FileNotFoundError(
        f"Embedding file not found:\n{EMBEDDING_PATH}"
    )

test_embeddings = torch.load(
    EMBEDDING_PATH,
    map_location="cpu",
    weights_only=True
).float()

print(f"Embeddings loaded: {test_embeddings.shape}")

# Only the first 3957 metadata rows have matching embeddings
embedding_count = len(test_embeddings)

metadata = metadata.iloc[
    :embedding_count
].reset_index(drop=True)

print(
    f"Usable prediction records: {len(metadata)}"
)

# ============================================================
# LOAD FSSM
# ============================================================

print("\nLoading Balanced FSSM model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Balanced model not found:\n{MODEL_PATH}"
    )

fssm_model = FSSMModel(
    input_dim=768,
    hidden_dim=300,
    state_size=16,
    num_classes=5
)

fssm_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

fssm_model = fssm_model.to(DEVICE)
fssm_model.eval()

print("Balanced FSSM model loaded successfully.")

# ============================================================
# FIND DRUG PAIR
# ============================================================

def find_drug_pair(drug1, drug2):

    drug1 = drug1.strip().lower()
    drug2 = drug2.strip().lower()

    result = metadata[
        (
            (metadata["drug1"].astype(str).str.lower() == drug1)
            &
            (metadata["drug2"].astype(str).str.lower() == drug2)
        )
        |
        (
            (metadata["drug1"].astype(str).str.lower() == drug2)
            &
            (metadata["drug2"].astype(str).str.lower() == drug1)
        )
    ]

    return result

# ============================================================
# GET EMBEDDING
# ============================================================

def get_embedding(row_index):

    embedding = test_embeddings[row_index]

    return embedding.unsqueeze(0).to(DEVICE)

# ============================================================
# PREDICTION
# ============================================================

def predict_drug_pair(drug1, drug2):

    matches = find_drug_pair(drug1, drug2)

    if len(matches) == 0:

        return {
            "success": False,
            "error": (
                "This drug pair was not found "
                "in the current dataset."
            )
        }

    row = matches.iloc[0]

    row_index = row.name

    text = str(row["text"])

    embedding = get_embedding(row_index)

    with torch.no_grad():

        outputs = fssm_model(embedding)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    confidence = (
        probabilities[0, predicted_class].item()
        * 100
    )

    actual_label = int(row["label"])
    actual_type = str(row["type"])

    interaction_detected = (
        predicted_class != 0
    )

    if interaction_detected:

        prediction_text = "Interaction Detected"
        result_type = "interaction"

        explanation = (
            "The FSSM model identified "
            "an interaction-related pattern "
            "in the biomedical description "
            "of this drug pair."
        )

    else:

        prediction_text = "No Interaction Detected"
        result_type = "no-interaction"

        explanation = (
            "The FSSM model did not identify "
            "a significant interaction pattern "
            "in the biomedical description "
            "of this drug pair."
        )

    class_probabilities = []

    for class_id in range(5):

        probability = (
            probabilities[0, class_id].item()
            * 100
        )

        class_probabilities.append({
            "class_id": class_id,
            "name": CLASS_NAMES[class_id],
            "probability": round(
                probability,
                2
            )
        })

    return {
        "success": True,

        "drug1": str(row["drug1"]),
        "drug2": str(row["drug2"]),

        "prediction": prediction_text,
        "result_type": result_type,

        "predicted_class": predicted_class,

        "predicted_class_name":
            CLASS_NAMES[predicted_class],

        "confidence": round(
            confidence,
            2
        ),

        "actual_label": actual_label,
        "actual_type": actual_type,

        "dataset_text": text,

        "explanation": explanation,

        "probabilities":
            class_probabilities
    }

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "success": False,
                "error": "No input data received."
            })

        drug1 = str(
            data.get("drug1", "")
        ).strip()

        drug2 = str(
            data.get("drug2", "")
        ).strip()

        if not drug1 or not drug2:

            return jsonify({
                "success": False,
                "error":
                    "Please enter both drug names."
            })

        result = predict_drug_pair(
            drug1,
            drug2
        )

        return jsonify(result)

    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )
