from datasets import load_dataset


# Load IMDb dataset using the correct Hugging Face repo name.
# The repo is "stanfordnlp/imdb" rather than just "imdb".
dataset = load_dataset("stanfordnlp/imdb")

train_data = dataset["train"]
test_data = dataset["test"]


print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))


# Show one sample
sample = train_data[0]
label = sample["label"]
text = sample["text"]

print("\nLabel:", label)
print("Review:", text[:500])