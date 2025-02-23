import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify
from model_pipeline import retrain_model  # Import retrain function

app = Flask(__name__)

# Define directories
DATA_DIR = "data/"
MODEL_DIR = "models/"
RETRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_retrained.pkl")

# Load initial model and scaler
model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
model1 = joblib.load(os.path.join(MODEL_DIR, "xgboost_retrained.pkl"))
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))

def configure_routes(app):
    """ Configures all Flask routes for the application """
    
    @app.route('/')
    @app.route('/home')
    def home():
        return render_template('home.html')

    @app.route('/team')
    def team():
        return render_template('team.html')

    @app.route('/xgb')
    def xgb_open():
        return render_template('xgboost.html')

    @app.route('/xgbretrain')
    def xgbretrain():
        return render_template('retrain.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        model_name = request.form.get('model_selector')

        # Get input from form
        input_data = {
            'International plan': [int(request.form['international_plan'])],
            'Number vmail messages': [float(request.form['number_vmail_messages'])],
            'Total day minutes': [float(request.form['total_day_minutes'])],
            'Total eve minutes': [float(request.form['total_eve_minutes'])],
            'Total night minutes': [float(request.form['total_night_minutes'])],
            'Total intl minutes': [float(request.form['total_intl_minutes'])],
            'Total intl calls': [int(request.form['total_intl_calls'])],
            'Customer service calls': [int(request.form['customer_service_calls'])]
        }

        # Convert to DataFrame and scale
        input_df = pd.DataFrame(input_data)
        input_df = scaler.transform(input_df)

        try:
            # Make prediction based on the selected model
            if model_name == 'xgboost':
                prediction = model.predict(input_df)
            elif model_name == 'xgboost1':
                prediction = model1.predict(input_df)
            else:
                return jsonify({'result': 'Invalid model selection!'}), 400

            # Determine the result based on prediction
            result = "The customer will not churn." if prediction == 0 else "The customer is likely to churn."
            return jsonify({'result': result})

        except Exception as e:
            return jsonify({'error': str(e), 'success': False})

    @app.route('/retrain', methods=['POST'])
    def retrain():
        """Retrains the model with user-specified hyperparameters"""
        try:
            # Check if preprocessed data exists
            data_files = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
            missing_files = [f for f in data_files if not os.path.exists(os.path.join(DATA_DIR, f))]

            if missing_files:
                return jsonify({'message': 'No prepared data found!', 'success': False})

            # Load the preprocessed datasets
            X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
            X_test = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
            y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
            y_test = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

            # Extract hyperparameters from request
            params = request.json
            learning_rate = float(params.get("learning_rate", 0.1))
            max_depth = int(params.get("max_depth", 3))
            n_estimators = int(params.get("n_estimators", 100))
            subsample = float(params.get("subsample", 1.0))
            colsample_bytree = float(params.get("colsample_bytree", 1.0))
            gamma = float(params.get("gamma", 0))
            min_child_weight = float(params.get("min_child_weight", 1))

            # Retrain the model
            model, best_params, metrics = retrain_model(
                X_train, X_test, y_train, y_test,
                learning_rate=learning_rate, max_depth=max_depth,
                n_estimators=n_estimators, subsample=subsample,
                colsample_bytree=colsample_bytree, gamma=gamma,
                min_child_weight=min_child_weight,
                retrained_model_path=RETRAINED_MODEL_PATH
            )

            if model is None:
                return jsonify({'message': 'Training failed.', 'success': False})

            return jsonify({
                'message': 'Model retrained successfully!',
                'accuracy': metrics.get("accuracy", 0),
                'success': True
            })

        except Exception as e:
            return jsonify({'message': str(e), 'success': False})

if __name__ == "__main__":
    configure_routes(app)
    app.run(debug=True)

