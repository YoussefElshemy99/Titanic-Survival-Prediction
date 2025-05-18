from src.data_preprocessing import load_and_preprocess_data
from src.model_training import (
    knn_classifier,
    naive_bayes_classifier,
    logistic_regression_classifier,
    decision_tree_classifier,
    random_forest_classifier
)
from src.prediction import predict_new_passengers, save_predictions

def main():
    # Initialize variables to store models and their accuracies
    models = {
        'KNN': None,
        'Naive Bayes': None,
        'Logistic Regression': None,
        'Decision Tree': None,
        'Random Forest': None
    }
    accuracies = {}
    best_model = None
    best_accuracy = 0
    data_loaded = False
    
    while True:
        print("\n=== Titanic Survival Prediction System ===")
        print("1. Train Models")
        print("2. Make Predictions")
        print("3. View Model Accuracies")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Load and preprocess data
            print("\nLoading and preprocessing data...")
            features_train, features_test, target_train, target_test, pred_data = load_and_preprocess_data()
            data_loaded = True
            
            # Train all models
            print("Training models...")
            
            # KNN
            models['KNN'], accuracies['KNN'] = knn_classifier(
                features_train, features_test, target_train, target_test
            )
            
            # Naive Bayes
            models['Naive Bayes'], accuracies['Naive Bayes'] = naive_bayes_classifier(
                features_train, features_test, target_train, target_test
            )
            
            # Logistic Regression
            models['Logistic Regression'], accuracies['Logistic Regression'] = logistic_regression_classifier(
                features_train, features_test, target_train, target_test
            )
            
            # Decision Tree
            models['Decision Tree'], accuracies['Decision Tree'] = decision_tree_classifier(
                features_train, features_test, target_train, target_test
            )
            
            # Random Forest
            models['Random Forest'], accuracies['Random Forest'] = random_forest_classifier(
                features_train, features_test, target_train, target_test
            )
            
            # Find best model
            best_model = max(accuracies.items(), key=lambda x: x[1])[0]
            best_accuracy = accuracies[best_model]
            
            print("\nAll models have been trained successfully!")
            
        elif choice == '2':
            if not data_loaded:
                print("\nError: Please train the models first (Option 1)!")
                continue
                
            print(f"\nUsing the best model ({best_model}) with accuracy: {best_accuracy:.2%}")
            
            # Load the prediction data again to ensure we have fresh data
            _, _, _, _, pred_data = load_and_preprocess_data()
            
            # Make predictions using the best model
            results = predict_new_passengers(models[best_model], pred_data)
            
            # Save predictions
            save_predictions(results)
            print("Predictions have been made and saved to the results folder.")
            
        elif choice == '3':
            if not data_loaded:
                print("\nError: Please train the models first (Option 1)!")
                continue
                
            print("\nModel Accuracies:")
            print("-----------------")
            for model, accuracy in accuracies.items():
                print(f"{model}: {accuracy:.2%}")
            print(f"\nBest Model: {best_model} with accuracy: {best_accuracy:.2%}")
            
        elif choice == '4':
            print("\nThank you for using the Titanic Survival Prediction System!")
            break
            
        else:
            print("\nInvalid choice! Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
