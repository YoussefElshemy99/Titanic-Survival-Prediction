import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model on test data
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def compare_models_metrics(evaluation_results):
    # Extract data for comparison
    model_names = [result['model_name'] for result in evaluation_results]
    accuracies = [result['accuracy'] for result in evaluation_results]

    # Print comparison
    print("\nModel Comparison:")
    for model, acc in zip(model_names, accuracies):
        print(f"{model}: {acc:.2f}")

    # Find best model
    best_idx = np.argmax(accuracies)
    best_model = evaluation_results[best_idx]['model']
    print(f"\nBest Model: {model_names[best_idx]} with accuracy {accuracies[best_idx]:.2f}")

    return best_model