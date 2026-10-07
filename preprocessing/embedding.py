import torch
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def load_pubmedbert():
    print("Loading PubMedBERT...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    model.eval()

    print("PubMedBERT loaded successfully.")

    return tokenizer, model


def create_embedding(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS representation
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding


if __name__ == "__main__":

    tokenizer, model = load_pubmedbert()

    sample_text = (
        "drug1 interaction with drug2 "
        "may increase the risk of adverse effects"
    )

    embedding = create_embedding(
        sample_text,
        tokenizer,
        model
    )

    print("\nEmbedding shape:")
    print(embedding.shape)

    print("\nEmbedding dimension:")
    print(embedding.shape[-1])