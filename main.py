import argparse
import pandas as pd
import shap
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    explain_customer_churn_with_message
)


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline for Customer Churn Prediction")
    
    parser.add_argument('--prepare', action='store_true', help='Prepare the data only')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--explain', type=int, help='Explain churn for a specific customer by index')
    
    parser.add_argument('--train_path', type=str, help='Path to the training dataset', required=True)
    parser.add_argument('--test_path', type=str, help='Path to the test dataset', required=True)
    parser.add_argument('--model_path', type=str, help='Path to save/load the model', default="models/xgboost_model.pkl")

    args = parser.parse_args()

    # Step 1: Prepare Data
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(args.train_path, args.test_path)

    if args.prepare:
        print("Data preparation completed.")
        return

    # Step 2: Train Model
    model = None
    if args.train:
        print("Training model...")
        model, best_params = train_model(X_train, y_train)
        save_model(model, args.model_path)
        print("Model training completed and saved.")

    # Step 3: Load Model (if not trained in this run)
    if not model:
        print("Loading saved model...")
        model = load_model(args.model_path)

    if not model:
        print("No trained model available. Train a model first.")
        return

    # Step 4: Evaluate Model
    if args.evaluate:
        print("Evaluating model...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        print("Model evaluation results:", metrics)

    # Step 5: Explain a Customer's Churn Prediction
    if args.explain is not None:
        print("Explaining customer churn...")
        
        feature_columns = [
            'International plan', 
            'Number vmail messages', 
            'Total day minutes',
            'Total eve minutes', 
            'Total night minutes', 
            'Total intl minutes',
            'Total intl calls', 
            'Customer service calls'
        ]

        if args.explain < 0 or args.explain >= len(X_test):
            print(f"Invalid customer index. Choose a number between 0 and {len(X_test) - 1}.")
            return

        explainer_xgb = shap.Explainer(model, X_train)
        specific_customer_df = pd.DataFrame([X_test[args.explain]], columns=feature_columns)

        result = explain_customer_churn_with_message(
            customer_data=specific_customer_df,
            model=model,
            explainer=explainer_xgb,
            feature_columns=feature_columns
        )

        print("\n=== Customer Churn Explanation ===")
        print("Prediction:", result["Prediction"])
        print("Churn Probability:", result["Churn Probability"])
        print("Explanation:", result["Explanation"])


if __name__ == "__main__":
    main()

