import pandas as pd

large = pd.read_csv("english_19.7k.tsv", sep="\t")
original = pd.read_parquet("data/en-00000-of-00001.parquet")

large_text = set(
    large["toxic"]
    .astype(str)
    .str.strip()
)

original_text = set(
    original["toxic_sentence"]
    .astype(str)
    .str.strip()
)

overlap = large_text & original_text

print("Large dataset toxic sentences:", len(large_text))
print("Original English toxic sentences:", len(original_text))
print("Exact overlap:", len(overlap))

if overlap:
    print("\nExamples of overlap:")
    for text in list(overlap)[:10]:
        print("-", text)