import joblib

def load_model():
    model = joblib.load('/models/xgboost_model.pkl')
    return model

def predict(model, features):
    return model.predict([features])
