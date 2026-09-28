import re
import torch
import torch.nn as nn


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. Load Trained Model
# ============================================================

checkpoint = torch.load(
    "lstm_sentiment.pth",
    map_location=device,
    weights_only=False
)

vocab = checkpoint["vocab"]


# ============================================================
# 3. LSTM Model
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            2
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(
            embedded
        )

        last_hidden = hidden[-1]

        output = self.fc(last_hidden)

        return output


# ============================================================
# 4. Create Model
# ============================================================

model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=1
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("\nModel loaded successfully!")


# ============================================================
# 5. Text Cleaning
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 6. Encode Text
# ============================================================

MAX_LENGTH = 200


def encode_text(text):

    text = clean_text(text)

    tokens = text.split()

    encoded = []

    for word in tokens:

        if word in vocab:
            encoded.append(vocab[word])
        else:
            encoded.append(vocab["<UNK>"])

    # Truncate
    encoded = encoded[:MAX_LENGTH]

    # Padding
    while len(encoded) < MAX_LENGTH:
        encoded.append(vocab["<PAD>"])

    return encoded


# ============================================================
# 7. Prediction Function
# ============================================================

def predict_sentiment(review):

    encoded = encode_text(review)

    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    negative_probability = probabilities[0][0].item()
    positive_probability = probabilities[0][1].item()

    print("\n====================================")
    print("SENTIMENT PREDICTION")
    print("====================================")

    print("Review:", review)

    print(
        f"Negative: {negative_probability * 100:.2f}%"
    )

    print(
        f"Positive: {positive_probability * 100:.2f}%"
    )

    if prediction == 1:
        print("Prediction: Positive")
    else:
        print("Prediction: Negative")

    print("====================================")


# ============================================================
# 8. User Input
# ============================================================

review = input(
    "\nEnter a movie review: "
)

predict_sentiment(review)