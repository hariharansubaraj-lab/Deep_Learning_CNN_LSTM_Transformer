import re
import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import load_dataset


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# 2. LOAD IMDb DATASET
# ============================================================

dataset = load_dataset("stanfordnlp/imdb")

print("\nDataset loaded successfully!")
print(dataset)


# ============================================================
# 3. SELECT TRAINING DATA
# ============================================================

TRAIN_SAMPLES = 10000

train_data = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(TRAIN_SAMPLES))
)

print("\nTraining samples:", len(train_data))


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<br\s*/?>", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 5. BUILD VOCABULARY
# ============================================================

counter = Counter()

for item in train_data:

    text = clean_text(item["text"])

    tokens = text.split()

    counter.update(tokens)


VOCAB_SIZE = 15000

most_common_words = counter.most_common(VOCAB_SIZE - 2)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for idx, (word, _) in enumerate(most_common_words, start=2):

    vocab[word] = idx


print("\nVocabulary size:", len(vocab))


# ============================================================
# 6. ENCODE TEXT
# ============================================================

MAX_LENGTH = 300


def encode_text(text):

    text = clean_text(text)

    tokens = text.split()

    encoded = []

    for token in tokens:

        encoded.append(
            vocab.get(token, vocab["<UNK>"])
        )

    # Limit sequence length
    encoded = encoded[:MAX_LENGTH]

    actual_length = len(encoded)

    # Avoid zero length
    if actual_length == 0:
        encoded = [vocab["<UNK>"]]
        actual_length = 1

    # Padding
    if len(encoded) < MAX_LENGTH:

        encoded += [vocab["<PAD>"]] * (
            MAX_LENGTH - len(encoded)
        )

    return encoded, actual_length


# ============================================================
# 7. DATASET CLASS
# ============================================================

class ReviewDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        encoded, length = encode_text(
            item["text"]
        )

        input_tensor = torch.tensor(
            encoded,
            dtype=torch.long
        )

        length_tensor = torch.tensor(
            length,
            dtype=torch.long
        )

        label_tensor = torch.tensor(
            item["label"],
            dtype=torch.long
        )

        return (
            input_tensor,
            length_tensor,
            label_tensor
        )


# ============================================================
# 8. CREATE DATASET
# ============================================================

train_dataset = ReviewDataset(train_data)


# ============================================================
# 9. DATALOADER
# ============================================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nBatch size:", BATCH_SIZE)
print("Number of batches:", len(train_loader))


# ============================================================
# 10. LSTM MODEL
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
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            2
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        # Remove unnecessary PAD processing
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(
            packed
        )

        last_hidden = hidden[-1]

        output = self.fc(last_hidden)

        return output


# ============================================================
# 11. CREATE MODEL
# ============================================================

model = LSTMClassifier(
    vocab_size=len(vocab)
)

model = model.to(device)

print("\nLSTM Model:")
print(model)


# ============================================================
# 12. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 14. TRAINING
# ============================================================

EPOCHS = 3

print("\nStarting Training...")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for inputs, lengths, labels in train_loader:

        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(
            inputs,
            lengths
        )

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Prediction
        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = (
        correct / total
    ) * 100

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# ============================================================
# 15. SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab": vocab
    },
    "lstm_sentiment.pth"
)

print("\nTraining completed successfully!")

print(
    "Model saved as: lstm_sentiment.pth"
)