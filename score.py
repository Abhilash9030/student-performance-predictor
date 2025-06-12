import joblib
import json
import numpy as np
import os

def init():
    global classifier, regressor
    classifier = joblib.load("classifier.pkl")
    regressor = joblib.load("regressor.pkl")

def run(raw_data):
    data = json.loads(raw_data)
    X = np.array([data['math_score'], data['reading_score'], data['writing_score']]).reshape(1, -1)
    
    class_result = classifier.predict(X)[0]
    reg_result = regressor.predict(X)[0]
    
    return {
        "prediction": "Pass" if class_result == 1 else "Fail",
        "expected_score": round(reg_result, 2)
    }
