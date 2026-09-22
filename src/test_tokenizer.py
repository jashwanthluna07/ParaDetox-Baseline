from transformers import AutoTokenizer

MODEL_NAME = "google/mt5-small"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "You made a mistake you ass."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer(text)["input_ids"]

print("\nOriginal text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))