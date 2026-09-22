from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/mt5-detox"

print("Loading trained tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading trained model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Test sentences
test_sentences = [
    "You made a mistake you ass.",
    "My page should be protected first so that worthless scum like you can't keep vandalizing it.",
    "fuckin jerk get on my nerves.",
]

for text in test_sentences:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    )

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print("\n" + "=" * 60)
    print("INPUT : ", text)
    print("OUTPUT:", result)