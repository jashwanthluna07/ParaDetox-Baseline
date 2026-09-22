import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/mt5-detox-v2/checkpoint-5988"
TEST_FILE = "data/test_v2.csv"

print("Loading test data...")
df = pd.read_csv(TEST_FILE)

print("Test examples:", len(df))
print(df["language"].value_counts())

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small",
    use_fast=False
)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Device:", device)

def detoxify(text, language):
    prompt = f"detoxify {language}: {text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

# Show 10 English + 10 Hindi
for language in ["en", "hi"]:
    subset = df[df["language"] == language].head(10)

    print(f"\n\n===== {language.upper()} =====")

    for _, row in subset.iterrows():
        prediction = detoxify(
            row["toxic_sentence"],
            row["language"]
        )

        print("\nTOXIC:")
        print(row["toxic_sentence"])

        print("EXPECTED:")
        print(row["neutral_sentence"])

        print("MODEL:")
        print(prediction)

        print("-" * 70)