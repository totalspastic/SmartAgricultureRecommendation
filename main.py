from flask import Flask, render_template, request
import pickle
import numpy as np
# from myTraining import data_dict
from statistics import mode
from myTraining import scaler

dict = {20: 'rice', 11: 'maize', 3: 'chickpea', 9: 'kidneybeans', 18: 'pigeonpeas', 13: 'mothbeans', 14: 'mungbean', 2: 'blackgram', 10: 'lentil', 19: 'pomegranate',
        1: 'banana', 12: 'mango', 7: 'grapes', 21: 'watermelon', 15: 'muskmelon', 0: 'apple', 16: 'orange', 17: 'papaya', 4: 'coconut', 6: 'cotton', 8: 'jute', 5: 'coffee'}

app = Flask(__name__)
file = open("svm_model.pkl", "rb")
svm_model = pickle.load(file)
file.close()

file = open("nb_model.pkl", "rb")
nb_model = pickle.load(file)
file.close()

file = open("rf_model.pkl", "rb")
rf_model = pickle.load(file)
file.close()




def predictCrop(data):
    input_data_as_nparray = np.asarray(data)
    input_data_reshaped =input_data_as_nparray.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    svm_prediction = svm_model.predict(std_data)
    nb_prediction = nb_model.predict(std_data)
    rf_prediction = rf_model.predict(std_data)
    final_prediction = [mode([i,j,k]) for i,j,k in zip(svm_prediction, nb_prediction, rf_prediction)]

    return dict[final_prediction[0]]


list1 = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','PH','Rainfall']
list = []

@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        dict = (request.form)

        for i in list1:
            list.append(float(dict.getlist(i)[0]))
        Crop = predictCrop(list)
        list.clear()
        print(Crop)
        return render_template("index.html", prediction = Crop)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=False)
