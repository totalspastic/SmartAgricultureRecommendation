import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("Crop_recommendationnn.csv")

# Separating the data and labels
x = df.drop(columns="label", axis=1)
y = df["label"]

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
scaled_x = scaler.transform(x)

encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])

x = scaled_x
y = df["label"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size = .2, stratify = y, random_state = 1)

svm_model = SVC()
svm_model.fit(xtrain, ytrain)

nb_model = GaussianNB()
nb_model.fit(xtrain,ytrain)

rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(xtrain,ytrain)

file  = open("svm_model.pkl", "wb")
pickle.dump(svm_model, file)
file.close()

file  = open("nb_model.pkl", "wb")
pickle.dump(nb_model, file)
file.close()

file  = open("rf_model.pkl", "wb")
pickle.dump(rf_model, file)
file.close()