import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import mlflow
<<<<<<< HEAD
=======
import psutil  
import time
<<<<<<< HEAD
=======
import datetime 
import json
>>>>>>> bad1a52 (heyyy)
from elasticsearch import Elasticsearch

>>>>>>> 471c58c (added elastic search)
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

    with mlflow.start_run(run_name="Training the model"):
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
    
<<<<<<< HEAD
    mlflow.log_params(best_params_random)

    mlflow.sklearn.log_model(tuned_xgb_model, "model")

    print("Training phase of the model executed successfully!")
=======
        train_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", train_time)
    
        mlflow.log_params(best_params_random)

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.sklearn.log_model(tuned_xgb_model, "model")

        print("Training phase of the model executed successfully!")
    
        model_name = "XGBoost_Classifier"
        mlflow.register_model(model_uri, model_name)

        print(f"Model trained and registered as '{model_name}' in MLflow Model Registry.")
>>>>>>> 709219f (heeeyy)

    return tuned_xgb_model, best_params_random


def evaluate_model(model, X_train, X_test, y_train, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="binary")
    test_recall = recall_score(y_test, y_test_pred, average="binary")
    test_f1 = f1_score(y_test, y_test_pred, average="binary")

    with mlflow.start_run(run_name="Evaluating the model"):
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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
def retrain_model(X_train, X_test, y_train, y_test, params, model_path="xgb_retrained.pkl"):
=======
=======
es = Elasticsearch(["http://localhost:9200"]) 
=======
=======
>>>>>>> 5aade7e (corrected code in model_pipleine)
es = Elasticsearch(["http://localhost:5601"]) 
>>>>>>> 20f6dda (added kibana to visualize data)
index_name = "mlflow_logs" 

class ElasticsearchHandler(logging.Handler):
    def __init__(self, es_client, index_name):
        super().__init__()
        self.es = es_client
        self.index_name = index_name
=======

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_elasticsearch():
    try:
        es = Elasticsearch(['http://localhost:9200'])  # Ensure this matches your setup
        info = es.info()
        logger.info(f"Connected to Elasticsearch: {info['name']} (version: {info['version']['number']})")
        return es
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {str(e)}")
        return None
>>>>>>> bad1a52 (heyyy)

    def emit(self, record):
        log_entry = self.format(record)
        self.es.index(index=self.index_name, document={"log": log_entry})

es_handler = ElasticsearchHandler(es, index_name)
formatter = logging.Formatter('%(asctime)s - %(message)s')
es_handler.setFormatter(formatter)

logging.getLogger().addHandler(es_handler)
logging.getLogger().setLevel(logging.INFO)

<<<<<<< HEAD
>>>>>>> 471c58c (added elastic search)
=======
def log_to_elasticsearch(es, run_id, metrics, params, artifacts=None):
    if es is None:
        logger.warning("Elasticsearch connection not available. Skipping log_to_elasticsearch.")
        return

    timestamp = datetime.datetime.now().isoformat()
    doc = {
        "timestamp": timestamp,
        "run_id": run_id,
        "metrics": metrics,
        "params": params,
        "artifacts": artifacts or []
    }

    try:
        logger.info(f"Sending document to Elasticsearch: {json.dumps(doc, default=str)[:200]}...")
        res = es.index(index="mlflow-logs", document=doc)
        logger.info(f"Log indexed to Elasticsearch: {res['result']} (index: {res['_index']}, id: {res['_id']})")
    except Exception as e:
        logger.error(f"Error indexing to Elasticsearch: {str(e)}")
        
>>>>>>> 5aade7e (corrected code in model_pipleine)
def retrain_model(X_train, X_test, y_train, y_test, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0, colsample_bytree=1.0, gamma=0, min_child_weight=1, retrained_model_path ="xgb_retrained.pkl"):
>>>>>>> 055825c (added parameters for retraining)
    """
    Retrains the XGBoost model with given hyperparameters, saves it, and logs results in MLflow.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        params (dict): Hyperparameters for model training.
        model_path (str): Path to save the trained model.

    Returns:
        tuple: (Trained model, best parameters, evaluation metrics)
    """
    if X_train is None or X_test is None or y_train is None or y_test is None:
        print("No prepared data, please run the stage of prepare")
        return None, None, None  # Return early if no data is provided

    params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'random_state': 42
    }
    
<<<<<<< HEAD
<<<<<<< HEAD
    with mlflow.start_run(run_name="Retraining the model"):
=======
    with mlflow.start_run(run_name="Retraining the model", log_system_metrics=True) as run:
        run_id = run.info.run_id
>>>>>>> 144c6e4 (log system's information)
=======
    with mlflow.start_run(run_name="Retraining the model", log_system_metrics=True) as run:
        run_id = run.info.run_id
=======
<<<<<<< HEAD
    with mlflow.start_run(run_name="Retraining the model", log_system_metrics=True) as run2:
<<<<<<< HEAD
        run_id = run2.info.run_id
=======
        run_id2 = run2.info.run_id
=======
    es = init_elasticsearch()
    
    with mlflow.start_run(run_name="Retraining the model", log_system_metrics=True) as run:
        run_id = run.info.run_id
>>>>>>> bad1a52 (heyyy)
>>>>>>> 70481b9 (corrected code in model_pipleine)
>>>>>>> c91d67c (corrected code in model_pipleine)
>>>>>>> 5aade7e (corrected code in model_pipleine)
        model = xgb.XGBClassifier(**params, random_state=42)

        model.fit(X_train, y_train)

        joblib.dump(model, model_path)
        print(f"Retrained model saved to {model_path}")

        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, "retrained_model")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, average="binary"),
            "recall": recall_score(y_test, y_test_pred, average="binary"),
            "f1_score": f1_score(y_test, y_test_pred, average="binary"),
        }

        mlflow.log_metrics(metrics)

<<<<<<< HEAD
=======
<<<<<<< HEAD
        model_uri = f"runs:/{run2.info.run_id}/model"
        mlflow.sklearn.log_model(tuned_xgb_model, "model")
=======
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.sklearn.log_model(model, "model")
>>>>>>> 45daa6a (log system)

        logger.info("Retraining phase of the model executed successfully!")
        
        print("Reraining phase of the model executed successfully!")
    
        model_name = "XGBoost_Retrained"
        mlflow.register_model(model_uri, model_name)

<<<<<<< HEAD
=======
        # Log to Elasticsearch
        log_to_elasticsearch(es, run_id, metrics, params)

        logger.info(f"Model retrained and registered as '{model_name}' in MLflow Model Registry.")
        logger.info("\nEvaluation Metrics for Retrained Model:")
        for metric, value in metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.5f}")
        
>>>>>>> bad1a52 (heyyy)
        print(f"Model retrained and registered as '{model_name}' in MLflow Model Registry.")
>>>>>>> 144c6e4 (log system's information)
        print("\nEvaluation Metrics for Retrained Model:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.5f}")

        print("Retraining and evaluation completed successfully!")

    return model, params, metrics
    
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

