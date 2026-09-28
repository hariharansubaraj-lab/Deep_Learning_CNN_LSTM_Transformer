import re
import torch

from collections import Counter
from datasets import load_dataset


# ==========================================
# 1. Load IMDb
# ==========================================

dataset = load_dataset("stanfordnlp/imdb")


train_data = dataset["train"]
test_data = dataset["test"]


# ==========================================
# 2. Text Cleaning
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    return text


# ==========================================
# 3. Tokenization
# ==========================================

def tokenize(text):

    return clean_text(text).split()


# ==========================================
# 4. Build Vocabulary
# ==========================================

counter = Counter()

for item in train_data:

    tokens = tokenize(item["text"])

    counter.update(tokens)


vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}


for word, count in counter.most_common(10000):

    vocab[word] = len(vocab)


print("Vocabulary size:", len(vocab))


# ==========================================
# 5. Convert Text to Numbers
# ==========================================

def encode_text(text, max_length=200):

    tokens = tokenize(text)

    ids = [
        vocab.get(word, vocab["<UNK>"])
        for word in tokens
    ]

    ids = ids[:max_length]

    while len(ids) < max_length:
        ids.append(vocab["<PAD>"])

    return ids


# Test
sample = train_data[0]["text"]

encoded = encode_text(sample)

print("Encoded length:", len(encoded))

torch.tensor(encoded)