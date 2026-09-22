import pandas as pd

FILE = "english_19.7k.tsv"

df = pd.read_csv(FILE, sep="\t")

print("Original rows:", len(df))

print("\nAvailable neutral rewrites:")
print("neutral1:", df["neutral1"].notna().sum())
print("neutral2:", df["neutral2"].notna().sum())
print("neutral3:", df["neutral3"].notna().sum())

# Count total toxic -> neutral pairs
pair_count = (
    df["neutral1"].notna().sum()
    + df["neutral2"].notna().sum()
    + df["neutral3"].notna().sum()
)

print("\nTotal possible toxic -> neutral pairs:", pair_count)

# Check duplicate toxic sentences
print("\nDuplicate toxic sentences:")
print(df["toxic"].duplicated().sum())

# Check duplicate neutral rewrites within each row
same_12 = (df["neutral1"] == df["neutral2"]).sum()
same_13 = (df["neutral1"] == df["neutral3"]).sum()
same_23 = (df["neutral2"] == df["neutral3"]).sum()

print("\nDuplicate neutral rewrites:")
print("neutral1 == neutral2:", same_12)
print("neutral1 == neutral3:", same_13)
print("neutral2 == neutral3:", same_23)