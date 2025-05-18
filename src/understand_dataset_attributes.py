import pandas as pd

# Load the dataset
df = pd.read_csv("C:\\Users\\Youssef Elshemy\\PycharmProjects\\Titanic Survival Prediction\\data\\Titanic dataset.csv")

# 1. Basic info
print("🔹 Dataset Info:")
print(df.info())
print("\n")

# 2. First few rows
print("🔹 First 5 Rows:")
print(df.head())
print("\n")

# 3. Summary statistics
print("🔹 Summary Statistics:")
print(df.describe(include="all"))
print("\n")

# 4. Missing values
print("🔹 Missing Values:")
print(df.isnull().sum())
print("\n")

# 5. Unique values per column
print("🔹 Unique Values in Each Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
print("\n")