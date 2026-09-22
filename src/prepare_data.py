import pandas as pd

# Load English dataset
en = pd.read_parquet("data/en-00000-of-00001.parquet")
en["language"] = "en"

# Load Hindi dataset
hi = pd.read_parquet("data/hi-00000-of-00001.parquet")
hi["language"] = "hi"

# Combine both datasets
df = pd.concat([en, hi], ignore_index=True)

# Keep only the columns we need
df = df[["language", "toxic_sentence", "neutral_sentence"]]

print("Total examples:", len(df))

print("\nExamples by language:")
print(df["language"].value_counts())

print("\nFirst 10 examples:")
print(df.head(10).to_string())

# Save combined dataset
df.to_csv("data/multilingual_detox.csv", index=False)

print("\nSaved: data/multilingual_detox.csv")