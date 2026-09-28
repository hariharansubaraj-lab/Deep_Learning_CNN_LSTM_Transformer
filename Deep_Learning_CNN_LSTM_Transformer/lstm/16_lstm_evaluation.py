import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# 2. Load IMDb Dataset
# ============================================================

dataset = load_dataset("stanfordnlp/imdb")

# Shuffle test data before selecting samples
test_data = dataset["test"].shuffle(seed=42).select(range(5000))

print("\nTest samples:", len(test_data))


# ============================================================
# 3. Load Trained Model + Vocabulary
# ============================================================

checkpoint = torch.load(
    "lstm_sentiment.pth",
    map_location=device,
    weights_only=False
)

vocab = checkpoint["vocab"]


# ============================================================
# 4. Text Cleaning
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text)

    # Keep only alphabets and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 5. Encode Text
# ============================================================

MAX_LENGTH = 200


def encode_text(tokens):

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
# 6. Test Dataset
# ============================================================

class ReviewDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        text = clean_text(item["text"])

        tokens = text.split()

        encoded = encode_text(tokens)

        input_tensor = torch.tensor(
            encoded,
            dtype=torch.long
        )

        label_tensor = torch.tensor(
            item["label"],
            dtype=torch.long
        )

        return input_tensor, label_tensor


test_dataset = ReviewDataset(test_data)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ============================================================
# 7. LSTM Model
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

        # Word Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Classification layer
        self.fc = nn.Linear(
            hidden_dim,
            2
        )

    def forward(self, x):

        # Convert token IDs to embeddings
        embedded = self.embedding(x)

        # LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # Last hidden state
        last_hidden = hidden[-1]

        # Classification
        output = self.fc(last_hidden)

        return output


# ============================================================
# 8. Create Model
# ============================================================

model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=1
)

model = model.to(device)


# ============================================================
# 9. Load Trained Weights
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("\nModel loaded successfully!")


# ============================================================
# 10. Evaluation
# ============================================================

correct = 0
total = 0


with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Prediction
        outputs = model(inputs)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# ============================================================
# 11. Accuracy
# ============================================================

accuracy = 100 * correct / total


print("\n====================================")
print("LSTM MODEL EVALUATION")
print("====================================")

print("Test Samples:", total)

print(
    f"Correct Predictions: {correct}"
)

print(
    f"Wrong Predictions: {total - correct}"
)

print(
    f"LSTM Test Accuracy: {accuracy:.2f}%"
)

print("====================================")