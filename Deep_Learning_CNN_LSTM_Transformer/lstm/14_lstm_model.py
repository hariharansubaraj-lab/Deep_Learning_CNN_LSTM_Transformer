import torch
import torch.nn as nn


# ==========================================
# LSTM Sentiment Classification Model
# ==========================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        # ==========================================
        # Embedding Layer
        # ==========================================
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # ==========================================
        # LSTM Layer
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # ==========================================
        # Classification Layer
        # ==========================================
        self.fc = nn.Linear(
            hidden_dim,
            2
        )

    # ==========================================
    # Forward Pass
    # ==========================================
    def forward(self, x):

        # Convert word IDs into embeddings
        embedded = self.embedding(x)

        # Pass embeddings through LSTM
        output, (hidden, cell) = self.lstm(
            embedded
        )

        # Take final hidden state
        last_hidden = hidden[-1]

        # Classification
        output = self.fc(last_hidden)

        return output


# ==========================================
# Test the Model
# ==========================================

vocab_size = 10000

model = LSTMClassifier(
    vocab_size=vocab_size
)

print("LSTM Model:")
print(model)


# ==========================================
# Create Dummy Input
# ==========================================

dummy_input = torch.randint(
    0,
    vocab_size,
    (1, 200)
)

print("\nInput shape:", dummy_input.shape)


# ==========================================
# Forward Pass
# ==========================================

output = model(dummy_input)

print("Output shape:", output.shape)
print("Output:", output)