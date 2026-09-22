import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


# ============================================================
# 1. GPU CHECK
# ============================================================

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available!")


# ============================================================
# 2. CONFIG
# ============================================================

MODEL_NAME = "google/mt5-small"

TRAIN_FILE = "data/train_v2.csv"
VAL_FILE = "data/validation_v2.csv"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print("Training examples:", len(train_df))
print("Validation examples:", len(val_df))

print("\nTraining languages:")
print(train_df["language"].value_counts())

print("\nValidation languages:")
print(val_df["language"].value_counts())


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 5. LOAD MODEL
# ============================================================

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.config.use_cache = False


# ============================================================
# 6. TOKENIZATION
# ============================================================

def preprocess_function(examples):

    inputs = [
        f"detoxify {lang}: {text}"
        for lang, text in zip(
            examples["language"],
            examples["toxic_sentence"]
        )
    ]

    targets = examples["neutral_sentence"]

    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True
    )

    labels = tokenizer(
        text_target=targets,
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing datasets...")

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)


# ============================================================
# 7. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


# ============================================================
# 8. GPU SMOKE TEST CONFIG
# ============================================================

training_args = Seq2SeqTrainingArguments(
    output_dir="models/mt5-detox-v2",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=5e-5,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=50,

    predict_with_generate=False,

    fp16=False,

    gradient_checkpointing=True,

    optim="adafactor",

    max_grad_norm=1.0,

    save_total_limit=2,

    report_to="none",
)

# ============================================================
# 9. TRAINER
# ============================================================

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_val,

    processing_class=tokenizer,

    data_collator=data_collator,
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\n========================================")
print("STARTING GPU SMOKE TEST")
print("========================================\n")

trainer.train()


print("\n========================================")
print("GPU SMOKE TEST COMPLETE")
print("========================================")

print("\nFinal training metrics:")
print(trainer.state.log_history[-1])