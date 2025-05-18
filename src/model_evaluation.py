import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print results
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Return dictionary with metrics (for model comparison)
    return {
        'model_name': model_name,
        'accuracy': acc,
        'y_pred': y_pred
    }

def compare_models_metrics(evaluation_results):
    """
    Compare multiple models based on their accuracy
    """
    # Extract data for comparison
    model_names = [result['model_name'] for result in evaluation_results]
    accuracies = [result['accuracy'] for result in evaluation_results]
    
    # Print comparison
    print("\nModel Comparison:")
    for model, acc in zip(model_names, accuracies):
        print(f"{model}: {acc:.2f}")
    
    # Find best model
    best_idx = np.argmax(accuracies)
    print(f"\nBest Model: {model_names[best_idx]} with accuracy {accuracies[best_idx]:.2f}")
    
    return pd.DataFrame({'Accuracy': accuracies}, index=model_names)

def save_model(model, model_name):
    """
    Save trained model to disk
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    
    with open(f'results/{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"{model_name} model saved to disk")
