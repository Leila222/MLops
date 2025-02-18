import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    roc_curve,
    auc,
    make_scorer,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import joblib
import os
import shap


def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(X_train, os.path.join(output_dir, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(output_dir, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(output_dir, "y_train.pkl"))
    joblib.dump(y_test, os.path.join(output_dir, "y_test.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    print(f"Preprocessed data saved in {output_dir}")


def load_preprocessed_data(output_dir="data"):
    try:
        X_train = joblib.load(os.path.join(output_dir, "X_train.pkl"))
        X_test = joblib.load(os.path.join(output_dir, "X_test.pkl"))
        y_train = joblib.load(os.path.join(output_dir, "y_train.pkl"))
        y_test = joblib.load(os.path.join(output_dir, "y_test.pkl"))
        scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
        print(f"Preprocessed data loaded from {output_dir}")
        return X_train, X_test, y_train, y_test, scaler
    except FileNotFoundError:
        print("Preprocessed data not found. Running data preparation.")
        return None


def encode_categorical_features(data):
    encoded_data = data.copy()
    label_encoders = {}

    binary_features = ["International plan", "Voice mail plan", "Churn"]
    for feature in binary_features:
        le = LabelEncoder()
        encoded_data[feature] = le.fit_transform(encoded_data[feature])
        label_encoders[feature] = le

    le_state = LabelEncoder()
    encoded_data["State"] = le_state.fit_transform(encoded_data["State"])
    label_encoders["State"] = le_state

    return encoded_data, label_encoders


def cap_outliers(data, col_num):
    for col_ in col_num:
        Q1 = data[col_].quantile(0.25)
        Q3 = data[col_].quantile(0.75)

        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        data[col_] = data[col_].clip(lower=lower_bound, upper=upper_bound)

    return data


def prepare_data(train_path, test_path, output_dir="data", drop_columns=None, p_value_threshold=0.05):
    preprocessed_data = load_preprocessed_data(output_dir)
    if preprocessed_data:
        return preprocessed_data  

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if drop_columns is None:
        drop_columns = [
            "Total day charge",
            "Total eve charge",
            "Total night charge",
            "Total intl charge",
            "Voice mail plan",
        ]

    col_num = [
        "Account length",
        "Number vmail messages",
        "Total day minutes",
        "Total day calls",
        "Total day charge",
        "Total eve minutes",
        "Total eve calls",
        "Total eve charge",
        "Total night minutes",
        "Total night calls",
        "Total night charge",
        "Total intl minutes",
        "Total intl calls",
        "Total intl charge",
        "Customer service calls",
    ]

    train_data = cap_outliers(train_data, col_num)
    test_data = cap_outliers(test_data, col_num)

    train_data, label_encoders = encode_categorical_features(train_data)
    test_data, _ = encode_categorical_features(test_data)

    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)

    X_train = train_data.drop(columns=["Churn"])
    y_train = train_data["Churn"]
    X_test = test_data.drop(columns=["Churn"])
    y_test = test_data["Churn"]

    x_log = sm.add_constant(X_train)
    reg_log = sm.Logit(y_train, x_log)
    results_log = reg_log.fit(disp=0)

    significant_features = results_log.pvalues[
        results_log.pvalues < p_value_threshold
    ].index
    if "const" in significant_features:
        significant_features = significant_features.drop("const")

    if significant_features.empty:
        raise ValueError(
            "No significant features found based on the p-value threshold."
        )

    X_train = X_train[significant_features]
    X_test = X_test[significant_features]

    scaler = StandardScaler()
    X_train_st = scaler.fit_transform(X_train)
    X_test_st = scaler.transform(X_test)

    save_preprocessed_data(X_train_st, X_test_st, y_train, y_test, scaler, output_dir)
    print("Data preparation successful!")

    return X_train_st, X_test_st, y_train, y_test, scaler


def train_model(X_train_st, y_train):
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "min_child_weight": [1, 3, 5],
    }

    with mlflow.start_run():
        xgb_model = xgb.XGBClassifier(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=100,
            scoring="f1",
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

    random_search.fit(X_train_st, y_train)

    best_params_random = random_search.best_params_

    tuned_xgb_model = xgb.XGBClassifier(**best_params_random, random_state=42)

    tuned_xgb_model.fit(X_train_st, y_train)
    
    mlflow.log_params(best_params_random)

    mlflow.sklearn.log_model(tuned_xgb_model, "model")

    print("Training phase of the model executed successfully!")

    return tuned_xgb_model, best_params_random


def evaluate_model(model, X_train, X_test, y_train, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="binary")
    test_recall = recall_score(y_test, y_test_pred, average="binary")
    test_f1 = f1_score(y_test, y_test_pred, average="binary")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", test_accuracy)
        mlflow.log_metric("precision", test_precision)
        mlflow.log_metric("recall", test_recall)
        mlflow.log_metric("f1_score", test_f1)
        
    print("\nTest Metrics for XGBoost After Tuning:")
    print(f"Accuracy: {test_accuracy:.5f}")
    print(f"Precision: {test_precision:.5f}")
    print(f"Recall: {test_recall:.5f}")
    print(f"F1-Score: {test_f1:.5f}")
    print("Evaluation phase of the model executed successfully!")

    return {
        "accuracy": test_accuracy,
        "precision": test_precision,
        "recall": test_recall,
        "f1_score": test_f1,
    }


def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved at: {filename}")


def load_model(filename):
    if os.path.exists(filename):
        model = joblib.load(filename)
        print(f"Model loaded from: {filename}")
        return model
    else:
        raise FileNotFoundError(f"File {filename} not found.")


def explain_customer_churn_with_message(
    customer_data, model, explainer, feature_columns, threshold=0.5
):
    """
    Explain why a specific customer would churn using SHAP values.
    """

    # Ensure the input customer_data is in DataFrame format
    if isinstance(customer_data, pd.Series):
        customer_data = customer_data.to_frame().T
    elif not isinstance(customer_data, pd.DataFrame):
        customer_data = pd.DataFrame([customer_data], columns=feature_columns)

    probability = model.predict_proba(customer_data)[:, 1][0]
    prediction = "Churn" if probability >= threshold else "No Churn"

    # Use SHAP to explain the prediction
    shap_values = explainer(customer_data)

    shap_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "SHAP Value": shap_values.values[0],
            "Feature Value": customer_data.iloc[0].values,
        }
    )

    shap_df = shap_df.sort_values(by="SHAP Value", key=abs, ascending=False)

    # Craft explanation based on SHAP values
    if prediction == "Churn":
        reasons = []
        for _, row in shap_df.head(
            3
        ).iterrows():  # Top 3 features affecting the decision
            feature = row["Feature"]

            # Customize the explanation based on the most important features
            if feature == "Customer service calls":
                reasons.append(f"frequent customer service calls")
            elif feature == "Total day minutes":
                reasons.append(f"high daytime usage")
            elif feature == "Total intl calls":
                reasons.append(f"many international calls")
            else:
                reasons.append(f"{feature}")

        explanation_message = f"The customer churned because of {', '.join(reasons)}."
    else:
        explanation_message = "The customer did not churn."

    result = {
        "Prediction": prediction,
        "Churn Probability": probability,
        "Explanation": explanation_message,
    }

    return result

