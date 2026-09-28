from transformers import pipeline


# ==========================================
# Create Classification Pipeline
# ==========================================

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# ==========================================
# Test Sentences
# ==========================================

texts = [
    "I really enjoyed this movie.",
    "This movie was boring and terrible."
]


# ==========================================
# Prediction
# ==========================================

results = classifier(texts)


# ==========================================
# Display
# ==========================================

for text, result in zip(texts, results):

    print("\nText:", text)

    print(
        "Label:",
        result["label"]
    )

    print(
        "Confidence:",
        f"{result['score']:.4f}"
    )