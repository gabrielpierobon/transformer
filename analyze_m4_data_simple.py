import pandas as pd
import numpy as np

# Read a small sample of the data
train_df = pd.read_csv('data/raw/Monthly-train.csv', nrows=5)
test_df = pd.read_csv('data/raw/Monthly-test.csv', nrows=5)

print("Train Dataset Shape:", train_df.shape)
print("Test Dataset Shape:", test_df.shape)

# Check the first series in test data
print("\nTest Dataset First Row (M1):")
print(test_df.iloc[0].tolist()[:20])  # First 20 values

# Check the first series in train data
print("\nTrain Dataset First Row (M1):")
print(train_df.iloc[0].tolist()[:20])  # First 20 values

# Count non-NaN values in the first series of train data
non_nan_count = train_df.iloc[0].notna().sum() - 1  # -1 for the series ID
print(f"\nNumber of non-NaN values in M1 train series: {non_nan_count}")

# Get the last 60 values of the first series
m1_values = train_df.iloc[0].dropna().tolist()[1:]  # Skip series ID
m1_last_60 = m1_values[-60:] if len(m1_values) >= 60 else m1_values
print(f"\nLast 60 values of M1 train series (showing first 10): {m1_last_60[:10]}")

# Check test data structure
print("\nTest data columns:", test_df.columns.tolist()[:20])
print("Number of forecast points in test data:", len(test_df.columns) - 1)  # -1 for series ID 