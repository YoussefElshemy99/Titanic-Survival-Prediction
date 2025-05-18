import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import os

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
    
    # Print metrics
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred
    }

def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """
    Plot confusion matrix for model predictions
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    plt.savefig(f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test, model_name="Model"):
    """
    Plot ROC curve for model predictions
    """
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        plt.savefig(f'results/roc_curve_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
        
        return roc_auc
    except:
        print(f"Could not generate ROC curve for {model_name} - predict_proba not available")
        return None

def compare_models(evaluation_results):
    """
    Compare multiple models side by side
    """
    # Extract data for plotting
    model_names = [result['model_name'] for result in evaluation_results]
    accuracies = [result['accuracy'] for result in evaluation_results]
    precisions = [result['precision'] for result in evaluation_results]
    recalls = [result['recall'] for result in evaluation_results]
    f1_scores = [result['f1'] for result in evaluation_results]
    
    # Create bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.subplot()
    
    for i, model_name in enumerate(model_names):
        offset = width * multiplier
        rects = ax.bar(x + offset, [accuracies[i], precisions[i], recalls[i], f1_scores[i]], 
                        width, label=model_name)
        multiplier += 1
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(model_names))
    ax.set_ylim(0, 1)
    
    plt.savefig('results/model_comparison.png')
    plt.close()

def save_model(model, model_name):
    """
    Save trained model to disk
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    
    with open(f'results/{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"{model_name} model saved to disk")

def load_model(model_name):
    """
    Load trained model from disk
    """
    with open(f'results/{model_name.lower().replace(" ", "_")}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return model
