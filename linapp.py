from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("linearRegression.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('linear.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
             input_var = float(request.form['yox'])
             model_prediction =  ml_model.predict([[input_var]])
             model_prediction = round(float(model_prediction),2)
        except  valueError:
            return "please check the input you have given"
    return render_template("linresult.html",prediction = model_prediction)

if __name__ == '__main__':
    app.run(debug="true")