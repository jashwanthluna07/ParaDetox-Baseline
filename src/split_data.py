import pandas as pd
from sklearn.model_selection import train_test_split

# Load combined dataset
df = pd.read_csv("data/multilingual_detox.csv")

print("Total dataset:", len(df))

# First split: 80% train, 20% temporary
train, temp = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["language"]
)

# Second split: divide the 20% equally into validation and test
val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["language"]
)

print("\nSplit sizes:")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print("\nTrain language distribution:")
print(train["language"].value_counts())

print("\nValidation language distribution:")
print(val["language"].value_counts())

print("\nTest language distribution:")
print(test["language"].value_counts())

# Save splits
train.to_csv("data/train.csv", index=False)
val.to_csv("data/validation.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("\nSaved:")
print("data/train.csv")
print("data/validation.csv")
print("data/test.csv")