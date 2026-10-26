
# Mini Project 1: Titanic Survival Prediction
# Data Cleaning Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------
# 1. Load the Titanic dataset
# ---------------------------------

input_file = Path("train.csv")
output_file = Path("titanic_cleaned.csv")

if not input_file.exists():
    raise FileNotFoundError(
        "train.csv not found. Download it from Kaggle "
        "and place it in the project folder."
    )

df = pd.read_csv(input_file)

print("Original Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ---------------------------------
# 2. Check missing values
# ---------------------------------

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ---------------------------------
# 3. Clean missing data
# ---------------------------------

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Fill missing Fare with median
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Cabin has many missing values.
# Preserve the column and mark missing entries.
df["Cabin"] = df["Cabin"].fillna("Unknown")

# ---------------------------------
# 4. Encode categorical columns
# ---------------------------------

# Encode Sex: male = 0, female = 1
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# One-hot encode Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    prefix="Embarked",
    dtype=int
)

# ---------------------------------
# 5. Visualize age distribution
# ---------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("Titanic Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()

plt.savefig("age_distribution.png", dpi=300)
plt.show()

# ---------------------------------
# 6. Verify cleaned data
# ---------------------------------

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

print("\nCleaned Dataset Preview:")
print(df.head())

# ---------------------------------
# 7. Export cleaned dataset
# ---------------------------------

df.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved to: {output_file.resolve()}")
print("Data cleaning completed successfully!")
