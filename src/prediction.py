def predict_new_passengers(model, new_data):
   # Parameters:
   # - model: Trained model
   # - new_data: DataFrame with passenger features
    
   # Returns:
   # - DataFrame with original data and prediction results
    
    predictions = model.predict(new_data)
    result_df = new_data.copy()
    result_df["Survived"] = predictions
    return result_df


def save_predictions(result_df, encoders=None, path="results\\predicted_survival.csv"):
    # Create a copy to avoid modifying the original DataFrame
    output_df = result_df.copy()

    if encoders:
        # Reverse encoding
        if 'Sex' in output_df.columns and 'sex_encoder' in encoders:
            le_sex = encoders['sex_encoder']
            output_df['Sex'] = le_sex.inverse_transform(output_df['Sex'].astype(int))

        if 'Embarked' in output_df.columns and 'embarked_encoder' in encoders:
            le_embarked = encoders['embarked_encoder']
            output_df['Embarked'] = le_embarked.inverse_transform(output_df['Embarked'].astype(int))

        # Reverse scaling
        if 'scaler' in encoders:
            scaler = encoders['scaler']
            scaled_cols = [col for col in ['Age', 'Fare'] if col in output_df.columns]
            if scaled_cols:
                output_df[scaled_cols] = scaler.inverse_transform(output_df[scaled_cols])

    # Convert Survived to more readable format
    if 'Survived' in output_df.columns:
        output_df['Survived'] = output_df['Survived'].map({0: 'No', 1: 'Yes'})

    output_df.to_csv(path, index=False)
    print(f"Predictions saved to {path} with human-readable categories")
