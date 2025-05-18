import pandas as pd
import numpy as np
import os
from src.model_evaluation import load_model
import joblib

def predict_survival(model, new_data):
    """
    Make predictions on new passenger data
    
    Parameters:
    - model: Trained model
    - new_data: DataFrame with passenger features
    
    Returns:
    - DataFrame with predictions
    """
    # Make predictions
    predictions = model.predict(new_data)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Survived': predictions
    })
    
    # Map numeric predictions to text for better readability
    results['Survival Status'] = results['Survived'].map({
        0: 'Did Not Survive',
        1: 'Survived'
    })
    
    return results

def predict_survival_proba(model, new_data):
    """
    Make probability predictions on new passenger data
    
    Parameters:
    - model: Trained model
    - new_data: DataFrame with passenger features
    
    Returns:
    - DataFrame with probabilities
    """
    try:
        # Get probability predictions
        proba = model.predict_proba(new_data)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Probability of Not Surviving': proba[:, 0],
            'Probability of Surviving': proba[:, 1]
        })
        
        return results
    except:
        print("This model doesn't support probability predictions")
        return None

def save_predictions(predictions, filename="titanic_predictions.csv"):
    """
    Save predictions to a CSV file
    
    Parameters:
    - predictions: DataFrame with predictions
    - filename: Output filename
    """
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save to CSV
    predictions.to_csv(f'results/{filename}', index=False)
    print(f"Predictions saved to results/{filename}")

def ensemble_prediction(models, new_data, model_names=None):
    """
    Create ensemble predictions using voting from multiple models
    
    Parameters:
    - models: List of trained models
    - new_data: DataFrame with passenger features
    - model_names: List of model names (optional)
    
    Returns:
    - DataFrame with ensemble predictions
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    
    # Get predictions from each model
    all_predictions = {}
    for model, name in zip(models, model_names):
        all_predictions[name] = model.predict(new_data)
    
    # Create DataFrame with all model predictions
    pred_df = pd.DataFrame(all_predictions)
    
    # Voting (majority)
    pred_df['Ensemble'] = pred_df.mode(axis=1)[0]
    
    # Map numeric predictions to text
    pred_df['Survival Status'] = pred_df['Ensemble'].map({
        0: 'Did Not Survive',
        1: 'Survived'
    })
    
    return pred_df
