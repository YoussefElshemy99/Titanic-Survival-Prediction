import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data():
    # Load the Data
    train_test_data = pd.read_csv("data\\Titanic dataset.csv")
    pred_data = pd.read_csv("data\\New passengers.csv")

    # Drop Unnecessary Columns
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    train_test_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    pred_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Handle Missing Values (Age and Fare)
    train_test_data["Age"].fillna(train_test_data["Age"].median(), inplace=True)
    train_test_data["Embarked"].fillna(train_test_data["Embarked"].mode()[0], inplace=True)
    train_test_data["Fare"].fillna(train_test_data["Fare"].median(), inplace=True)

    pred_data["Age"].fillna(train_test_data["Age"].median(), inplace=True)
    pred_data["Embarked"].fillna(train_test_data["Embarked"].mode()[0], inplace=True)
    pred_data["Fare"].fillna(train_test_data["Fare"].median(), inplace=True)

    # Replace infinite values with median
    train_test_data["Fare"].replace([np.inf, -np.inf], train_test_data["Fare"].median(), inplace=True)
    pred_data["Fare"].replace([np.inf, -np.inf], train_test_data["Fare"].median(), inplace=True)

    # Ensure data is in float64 format before scaling
    train_test_data["Fare"] = train_test_data["Fare"].astype('float64')
    pred_data["Fare"] = pred_data["Fare"].astype('float64')

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    train_test_data["Sex"] = le_sex.fit_transform(train_test_data["Sex"])
    train_test_data["Embarked"] = le_embarked.fit_transform(train_test_data["Embarked"])

    pred_data["Sex"] = le_sex.transform(pred_data["Sex"])
    pred_data["Embarked"] = le_embarked.transform(pred_data["Embarked"])

    # Separate features and target
    features = train_test_data.drop(columns=["Survived"])
    target = train_test_data["Survived"]

    # Scale float columns: Age and Fare
    scaler = StandardScaler()
    float_cols = ["Age", "Fare"]
    features[float_cols] = scaler.fit_transform(features[float_cols])
    pred_data[float_cols] = scaler.transform(pred_data[float_cols])

    # Train/test split
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    return features_train, features_test, target_train, target_test, pred_data
