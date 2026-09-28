from transformers import pipeline


# ==========================================
# Text Generation Pipeline
# ==========================================

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)


# ==========================================
# Input
# ==========================================

prompt = "Artificial intelligence is"


# ==========================================
# Generate Text
# ==========================================

result = generator(
    prompt,
    max_new_tokens=50,
    num_return_sequences=1
)


# ==========================================
# Display
# ==========================================

print("Prompt:")
print(prompt)

print("\nGenerated Text:")
print(
    result[0]["generated_text"]
)