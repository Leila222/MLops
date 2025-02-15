from flask import request,  render_template, jsonify
import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('data/scaler.pkl')

def configure_routes(app):
    @app.route('/')
    def default():
        return render_template('home.html')
    
    @app.route('/home')
    def home():
        return render_template('home.html')

    @app.route('/team')
    def team():
        return render_template('team.html')

    @app.route('/xgb')
    def xgb_open():
        return render_template('xgboost.html')
    
    
    @app.route('/predict', methods=['POST'])
    def predict():

        international_plan = int(request.form['international_plan'])
        number_vmail_messages = float(request.form['number_vmail_messages'])
        total_day_minutes = float(request.form['total_day_minutes'])
        total_eve_minutes = float(request.form['total_eve_minutes'])
        total_night_minutes = float(request.form['total_night_minutes'])
        total_intl_minutes = float(request.form['total_intl_minutes'])
        total_intl_calls = int(request.form['total_intl_calls'])
        customer_service_calls = int(request.form['customer_service_calls'])

        data = {
            'International plan': [international_plan],
            'Number vmail messages': [number_vmail_messages],
            'Total day minutes': [total_day_minutes],
            'Total eve minutes': [total_eve_minutes],
            'Total night minutes': [total_night_minutes],
            'Total intl minutes': [total_intl_minutes],
            'Total intl calls': [total_intl_calls],
            'Customer service calls': [customer_service_calls]
        }
        
        input_df = pd.DataFrame(data)
        
        input_df = scaler.transform(input_df)

        prediction = model.predict(input_df)
        
        if prediction == 0:
            result = "The customer will not churn."
        else:
            result = "The customer is likely to churn."        
        return jsonify({'result': result})
    

