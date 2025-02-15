import joblib

def predict(model, features):
    return model.predict([features])
