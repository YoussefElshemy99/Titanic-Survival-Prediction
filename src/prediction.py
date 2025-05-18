import pandas as pd
import os

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

def save_predictions(result_df, path="results/predicted_survival.csv"):
    """
    Save predictions to a CSV file
    
    Parameters:
    - result_df: DataFrame with predictions
    - path: Output path for the CSV file
    """
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        
    result_df.to_csv(path, index=False)
    print(f"Predictions saved to {path}")
