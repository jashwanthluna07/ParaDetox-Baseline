import pandas as pd

df = pd.read_parquet("hi-00000-of-00001.parquet")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string())