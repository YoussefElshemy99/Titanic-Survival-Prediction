from sklearn.metrics import accuracy_score
from src.understand_dataset_attributes import dataset_info
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import (
    knn_classifier,
    naive_bayes_classifier,
    logistic_regression_classifier,
    decision_tree_classifier,
    random_forest_classifier,
    svm_classifier
)
from src.model_evaluation import evaluate_model, compare_models_metrics
from src.prediction import predict_new_passengers, save_predictions


def main():
    # Initialize variables
    models = {}
    evaluation_results = []
    best_model = None
    features_train, features_test, target_train, target_test, pred_data, encoders = None, None, None, None, None, None
    data_loaded = False

    while True:
        print("\n=== Titanic Survival Prediction System ===")
        print("1. Dataset Information")
        print("2. Preprocess & Train Models")
        print("3. Evaluate and Compare Models")
        print("4. Predict and Save Results with Best Model")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            # Dataset Information
            print("\nLoading dataset information...")
            dataset_info()

        elif choice == "2":
            # Preprocess & Train Models
            print("\nPreprocessing data and training models...")
            try:
                features_train, features_test, target_train, target_test, pred_data, encoders = load_and_preprocess_data()
                data_loaded = True

                # Train all models
                models['KNN'] = knn_classifier(features_train, target_train)

                models['Naive Bayes'] = naive_bayes_classifier(features_train, target_train)

                models['Logistic Regression'] = logistic_regression_classifier(features_train,target_train)

                models['Decision Tree'] = decision_tree_classifier(features_train, target_train)

                models['Random Forest'] = random_forest_classifier(features_train, target_train)

                models['SVM'] = svm_classifier(features_train, target_train)

                print("\nAll models trained successfully!")

            except Exception as e:
                print(f"\nError during preprocessing/training: {str(e)}")

        elif choice == "3":
            # Evaluate and Compare Models
            if not data_loaded:
                print("\nPlease preprocess and train models first (Option 2)!")
                continue

            print("\nEvaluating models...")
            for name, model in models.items():
                # Evaluate each model
                evaluate_model(model, features_test, target_test, name)

                # Store results for comparison
                y_pred = model.predict(features_test)
                evaluation_results.append({
                    'model_name': name,
                    'accuracy': accuracy_score(target_test, y_pred),
                    'model': model
                })

            # Compare models and select best one
            best_model = compare_models_metrics(evaluation_results)

        elif choice == "4":
            # Predict and Save Results
            if not data_loaded:
                print("\nPlease preprocess and train models first (Option 2)!")
                continue

            if not best_model:
                print("\nNo best model selected yet. Please evaluate models first (Option 3)!")
                continue

            print("\nMaking predictions on new passengers...")
            try:
                results = predict_new_passengers(best_model, pred_data)
                save_predictions(results, encoders)
                print("\nPredictions saved successfully!")
            except Exception as e:
                print(f"\nError during prediction: {str(e)}")

        elif choice == "5":
            # Exit
            print("\nExiting the Titanic Survival Prediction System. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter a number between 1-5.")


if __name__ == "__main__":
    main()