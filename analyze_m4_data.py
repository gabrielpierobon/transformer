import pandas as pd

# Read the first few rows of each dataset
train_df = pd.read_csv('data/raw/Monthly-train.csv', nrows=5)
test_df = pd.read_csv('data/raw/Monthly-test.csv', nrows=5)

print("Train Dataset:")
print("Columns:", train_df.columns.tolist())
print("Shape:", train_df.shape)
print("First row:", train_df.iloc[0].tolist())
print("\nTest Dataset:")
print("Columns:", test_df.columns.tolist())
print("Shape:", test_df.shape)
print("First row:", test_df.iloc[0].tolist())

# Check how many values each series has in the training set
print("\nAnalyzing series lengths in training set...")
train_full = pd.read_csv('data/raw/Monthly-train.csv')
series_lengths = {}
for _, row in train_full.iterrows():
    series_id = row['V1']
    # Count non-NaN values after the series ID
    length = row.iloc[1:].notna().sum()
    series_lengths[series_id] = length

# Print statistics about series lengths
lengths = list(series_lengths.values())
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Average length: {sum(lengths)/len(lengths):.2f}")
print(f"Number of series: {len(series_lengths)}")

# Check a specific series (M1) to understand its structure
print("\nExample series M1:")
m1_train = train_full[train_full['V1'] == 'M1'].iloc[0].dropna().tolist()
print(f"M1 series length: {len(m1_train) - 1}")  # -1 for the series ID
print(f"M1 first few values: {m1_train[1:11]}")
print(f"M1 last few values: {m1_train[-10:]}") 