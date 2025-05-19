import pandas as pd

def dataset_info():
   # Load the dataset
   df = pd.read_csv("C:\\Users\\Youssef Elshemy\\PycharmProjects\\Titanic Survival Prediction\\data\\Titanic dataset.csv")

   # Basic info
   print("🔹 Dataset Info:")
   print(df.info())
   print("\n")

   # First few rows
   print("🔹 First 5 Rows:")
   print(df.head())
   print("\n")

   # Summary statistics
   print("🔹 Summary Statistics:")
   print(df.describe(include="all"))
   print("\n")

   # Missing values
   print("🔹 Missing Values:")
   print(df.isnull().sum())
   print("\n")

   # Unique values per column
   print("🔹 Unique Values in Each Column:")
   for col in df.columns:
      print(f"{col}: {df[col].nunique()} unique values")
   print("\n")