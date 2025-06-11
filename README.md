# ML Customer Churn Pipeline  

This project automates the end-to-end machine learning workflow to predict customer churn for a telecommunications company. It leverages Jenkins for CI/CD orchestration, MLflow for experiment tracking, and Flask for model deployment as an API.

## End-to-End Pipeline Overview
The pipeline is triggered automatically on code changes (via GitHub Webhooks) and includes the following stages:
1. **Checkout Code** - Clones the latest version of the repository from GitHub.
2. **Set up Environment** - Creates a Python virtual environment and installs all necessary dependencies.
3. **Prepare Data** - Loads and preprocesses the training and testing datasets.
4. **Train Model** - Trains a machine learning model to predict customer churn using relevant features.
5. **Evaluate Model** - Evaluates the model's performance using metrics like accuracy, precision, recall, and AUC. All experiments are tracked using MLflow.
6. **Deploy API** - Deploys the best-performing model as a Flask REST API on the specified server.

## Technologies Used
**Jenkins** – CI/CD automation

**MLflow** – Model tracking and versioning

**Flask** – RESTful API for model serving

**Python** (scikit-learn, pandas, etc.)

**GitHub Webhooks** – Triggering pipeline on code updates

## Features
- Automated retraining and deployment with every GitHub commit

- MLflow logging and UI for model comparison

- Lightweight and easy-to-deploy REST API

- Modular and maintainable pipeline structure

## Project Structure 
```
├── app/ # Flask application logic (routes, retrain, API endpoints) 
├── data/ # Training and testing datasets 
├── models/ # Saved and serialized models (.pkl files) 
├── static/ # Static files for the web app (CSS, JS) 
├── templates/ # HTML templates for the web UI 
├── Jenkinsfile # Jenkins pipeline configuration 
├── README.md # Project documentation 
├── app.py # Entry point for running the Flask app 
├── main.py # Script to trigger model retraining 
├── makefile # Build/automation tasks 
├── model.py # Model class or utility functions 
├── model_pipeline.py # Main ML pipeline logic 
├── requirements.txt # Python dependencies
``` 
