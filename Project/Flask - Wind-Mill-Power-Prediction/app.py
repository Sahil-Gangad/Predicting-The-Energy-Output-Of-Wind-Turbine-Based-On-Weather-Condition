import numpy as np 
from flask import Flask, request, jsonify, render_template
import joblib
import requests
import windApp

app = Flask(__name__)

loaded_model = joblib.load('power_prediction.sav')

def format_to_two_decimals(value):
    return f"{value:.2f}"

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict')
def predict():
    return render_template('index2.html')

@app.route('/predict/<city>')
def name(city):
    result = list(windApp.result(city))
    return jsonify(result)

@app.route('/predict_info/<city>')
def prediction(city):
    result = list(windApp.result(city))
    print(result[3])
    x_single = np.array([[result[3]]])

    prediction = loaded_model.predict(x_single)
    print(prediction)

    prediction = float(format_to_two_decimals(prediction[0]))

    print(prediction)

    response_data = {
        'prediction':prediction,
        'result': result
    }

    return jsonify(response_data)


if __name__ == '__main__':
    app.run()