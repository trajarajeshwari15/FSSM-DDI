import os
import sys

import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModel


# ============================================================
# PROJECT PATH
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
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "test.csv"
)

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
# LOAD DATASET
# ============================================================

print("=" * 60)
print("FSSM-DDI WEB APPLICATION")
print("=" * 60)

print()
print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset loaded: {len(df)} records"
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
# LOAD FSSM
# ============================================================

print()
print("Loading Balanced FSSM model...")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nBalanced model not found:\n{MODEL_PATH}\n"
        "\nPlease make sure "
        "fssm_ddi_model_balanced.pt "
        "exists inside the model folder."
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

    drug1 = drug1.strip().lower()
    drug2 = drug2.strip().lower()

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
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

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
# PREDICTION
# ============================================================

def predict_drug_pair(
    drug1,
    drug2
):

    matches = find_drug_pair(
        drug1,
        drug2
    )

    # --------------------------------------------------------
    # NOT FOUND
    # --------------------------------------------------------

    if len(matches) == 0:

        return {
            "success": False,
            "error": (
                "This drug pair was not found "
                "in the current dataset."
            )
        }


    # --------------------------------------------------------
    # GET FIRST MATCH
    # --------------------------------------------------------

    row = matches.iloc[0]

    text = str(
        row["text"]
    )


    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    embedding = create_embedding(
        text
    )


    # --------------------------------------------------------
    # FSSM PREDICTION
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
    # ACTUAL VALUES
    # --------------------------------------------------------

    actual_label = int(
        row["label"]
    )

    actual_type = str(
        row["type"]
    )


    # --------------------------------------------------------
    # BINARY RESULT
    # --------------------------------------------------------

    interaction_detected = (
        predicted_class != 0
    )


    if interaction_detected:

        prediction_text = (
            "Interaction Detected"
        )

        result_type = "interaction"

        explanation = (
            "The FSSM model identified "
            "an interaction-related pattern "
            "in the biomedical description "
            "of this drug pair."
        )

    else:

        prediction_text = (
            "No Interaction Detected"
        )

        result_type = "no-interaction"

        explanation = (
            "The FSSM model did not identify "
            "a significant interaction pattern "
            "in the biomedical description "
            "of this drug pair."
        )


    # --------------------------------------------------------
    # CLASS PROBABILITIES
    # --------------------------------------------------------

    class_probabilities = []

    for class_id in range(5):

        probability = (
            probabilities[
                0,
                class_id
            ].item()
            * 100
        )

        class_probabilities.append(
            {
                "class_id": class_id,
                "name": CLASS_NAMES[class_id],
                "probability": round(
                    probability,
                    2
                )
            }
        )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "success": True,

        "drug1": str(
            row["drug1"]
        ),

        "drug2": str(
            row["drug2"]
        ),

        "prediction": prediction_text,

        "result_type": result_type,

        "predicted_class": predicted_class,

        "predicted_class_name": CLASS_NAMES[
            predicted_class
        ],

        "confidence": round(
            confidence,
            2
        ),

        "actual_label": actual_label,

        "actual_type": actual_type,

        "dataset_text": text,

        "explanation": explanation,

        "probabilities": class_probabilities
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

        data = request.get_json()

        if not data:

            return jsonify(
                {
                    "success": False,
                    "error": "No input received."
                }
            )


        drug1 = str(
            data.get(
                "drug1",
                ""
            )
        ).strip()


        drug2 = str(
            data.get(
                "drug2",
                ""
            )
        ).strip()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not drug1:

            return jsonify(
                {
                    "success": False,
                    "error": "Please enter Drug 1."
                }
            )


        if not drug2:

            return jsonify(
                {
                    "success": False,
                    "error": "Please enter Drug 2."
                }
            )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        result = predict_drug_pair(
            drug1,
            drug2
        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "An error occurred while "
                    "processing the prediction."
                )
            }
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify(
        {
            "status": "online",
            "model": "FSSM-DDI",
            "device": str(DEVICE)
        }
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SERVER READY")
    print("=" * 60)

    print()
    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )