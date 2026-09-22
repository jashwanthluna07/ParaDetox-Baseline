import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD LARGE ENGLISH DATASET
# ============================================================

large = pd.read_csv("english_19.7k.tsv", sep=None, engine="python")

english_pairs = []

for _, row in large.iterrows():
    toxic = str(row["toxic"]).strip()

    seen_neutral = set()

    for column in ["neutral1", "neutral2", "neutral3"]:
        neutral = row[column]

        if pd.isna(neutral):
            continue

        neutral = str(neutral).strip()

        if not neutral:
            continue

        # Remove duplicate neutral rewrites
        if neutral in seen_neutral:
            continue

        seen_neutral.add(neutral)

        english_pairs.append({
            "language": "en",
            "toxic_sentence": toxic,
            "neutral_sentence": neutral
        })


# ============================================================
# 2. LOAD ORIGINAL ENGLISH DATASET
# ============================================================

original_en = pd.read_parquet(
    "data/en-00000-of-00001.parquet"
)

original_en["language"] = "en"

original_en = original_en.rename(
    columns={
        "toxic_sentence": "toxic_sentence",
        "neutral_sentence": "neutral_sentence"
    }
)

original_en = original_en[
    ["language", "toxic_sentence", "neutral_sentence"]
]


# ============================================================
# 3. LOAD ORIGINAL HINDI DATASET
# ============================================================

original_hi = pd.read_parquet(
    "data/hi-00000-of-00001.parquet"
)

original_hi["language"] = "hi"

original_hi = original_hi[
    ["language", "toxic_sentence", "neutral_sentence"]
]


# ============================================================
# 4. CONVERT LARGE ENGLISH DATASET TO DATAFRAME
# ============================================================

large_en = pd.DataFrame(english_pairs)


print("Large English pairs:", len(large_en))
print("Original English pairs:", len(original_en))
print("Original Hindi pairs:", len(original_hi))


# ============================================================
# 5. COMBINE EVERYTHING
# ============================================================

df = pd.concat(
    [
        large_en,
        original_en,
        original_hi
    ],
    ignore_index=True
)


# ============================================================
# 6. CLEAN TEXT
# ============================================================

df["toxic_sentence"] = (
    df["toxic_sentence"]
    .astype(str)
    .str.strip()
)

df["neutral_sentence"] = (
    df["neutral_sentence"]
    .astype(str)
    .str.strip()
)


# Remove completely identical toxic → neutral pairs
df = df.drop_duplicates(
    subset=[
        "language",
        "toxic_sentence",
        "neutral_sentence"
    ]
).reset_index(drop=True)


print("\nFinal dataset size:", len(df))

print("\nLanguage distribution:")
print(df["language"].value_counts())


# ============================================================
# 7. SAVE COMPLETE DATASET
# ============================================================

df.to_csv(
    "data/multilingual_detox_v2.csv",
    index=False
)


# ============================================================
# 8. SPLIT BY TOXIC SOURCE SENTENCE
# ============================================================
#
# IMPORTANT:
# We split toxic sentences BEFORE expanding them into
# multiple neutral rewrites.
#
# This prevents the same toxic sentence from appearing
# in both train and test.
#


sources = (
    df[
        ["language", "toxic_sentence"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


train_sources, temp_sources = train_test_split(
    sources,
    test_size=0.20,
    random_state=42,
    stratify=sources["language"]
)

val_sources, test_sources = train_test_split(
    temp_sources,
    test_size=0.50,
    random_state=42,
    stratify=temp_sources["language"]
)


# Create keys for fast matching
train_keys = set(
    zip(
        train_sources["language"],
        train_sources["toxic_sentence"]
    )
)

val_keys = set(
    zip(
        val_sources["language"],
        val_sources["toxic_sentence"]
    )
)

test_keys = set(
    zip(
        test_sources["language"],
        test_sources["toxic_sentence"]
    )
)


# ============================================================
# 9. ASSIGN EACH PAIR TO ITS SPLIT
# ============================================================

def get_split(row):
    key = (
        row["language"],
        row["toxic_sentence"]
    )

    if key in train_keys:
        return "train"

    if key in val_keys:
        return "validation"

    if key in test_keys:
        return "test"

    raise ValueError("Source sentence not found!")


df["split"] = df.apply(
    get_split,
    axis=1
)


# ============================================================
# 10. CREATE FINAL SPLITS
# ============================================================

train = df[df["split"] == "train"].drop(
    columns=["split"]
)

validation = df[df["split"] == "validation"].drop(
    columns=["split"]
)

test = df[df["split"] == "test"].drop(
    columns=["split"]
)


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\nSplit sizes:")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

print("\nTrain languages:")
print(train["language"].value_counts())

print("\nValidation languages:")
print(validation["language"].value_counts())

print("\nTest languages:")
print(test["language"].value_counts())


# ============================================================
# 12. SAVE SPLITS
# ============================================================

train.to_csv(
    "data/train_v2.csv",
    index=False
)

validation.to_csv(
    "data/validation_v2.csv",
    index=False
)

test.to_csv(
    "data/test_v2.csv",
    index=False
)


print("\nSaved:")
print("data/multilingual_detox_v2.csv")
print("data/train_v2.csv")
print("data/validation_v2.csv")
print("data/test_v2.csv")