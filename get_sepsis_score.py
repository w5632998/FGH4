#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib  #模型持久化

def get_sepsis_score(data, model):
    data_mean = np.array([
        84.58144, 97.19395, 36.97723, 123.7505, 82.4001,
        63.83056, 18.7265, 32.95766, -0.68992, 24.07548,
        0.554839, 7.378934, 41.02187, 92.65419, 260.2234,
        23.91545, 102.4837, 7.557531, 105.8279, 1.510699,
        1.836177, 136.9323, 2.64666, 2.05145, 3.544238,
        4.135528, 2.114059, 8.290099, 30.79409, 10.43083,
        41.23119, 11.44641, 287.3857, 196.0139, 62.00947,
        0.559269, 0.496571, 0.503429, -56.1251, 26.99499])
    data_mean = pd.DataFrame(data_mean.reshape(1, 40))
    data = pd.DataFrame(data)
    data = data.fillna(method='pad')
    data = data.fillna(method='bfill')  # 数据病人本身填充

    values = pd.concat([data_mean, data], axis=0)
    values = values.fillna(method='pad')  # 引入平均值，再填充


    values.drop(values.columns[[7, 9, 10,14, 16, 18, 20, 22,  26, 27, 32]], axis=1, inplace=True)


    x_test = values[-1:]
    prediction_probas = model.predict_proba(x_test)
    prediction_proba = prediction_probas[-1]
    labels = model.predict(x_test)
    label = labels[-1]
    if label == 1:
        score = max(prediction_proba)
    else:
        score = min(prediction_proba)

    return score, label

def load_sepsis_model():

    model = joblib.load('clf_Voting.pkl')

    return model
