from flask import Flask, render_template, jsonify, request
from flask_cors import cross_origin, CORS
import pickle
# import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()

def homePage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@cross_origin()

def predict():
    try:
        Pregnancies = float(request.form['Pregnancies'])
        Glucose_Level = float(request.form['Glucose Level'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])
        fileName = 'LogisticRegression_model_1.sav'
        # model = pickle.load(open(fileName, 'rb'))
        model = pickle.load(open(fileName, 'rb'))
        # print(model)
        fileName_st = 'standardscaler_1.sav'
        standardScaler = pickle.load(open(fileName_st, 'rb'))
        try:
            st_data = standardScaler.transform([[Pregnancies, Glucose_Level, BloodPressure, SkinThickness,
                                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
            print(st_data)

        except Exception as e:
            print(e)
        try:
            model_prediction = model.predict(st_data)
        except Exception as e:
            print(e)
        print("123456")
        if model_prediction == 1:
            msg = 'You have Diabetes, please consult a Doctor.'
            print(model_prediction)
        else:
            msg = "You don't have Diabetes."
        print(msg)
        return render_template('results.html', prediction='{}'.format(msg))

    except Exception as e:
        return 'Somthing wrong'


if __name__ == "__main__":
    app.run(debug=True)
