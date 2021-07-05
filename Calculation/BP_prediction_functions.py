import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import Calculation.key_point_calculation
import Calculation.feature_calculation
import pandas as pd
from scipy import stats
import pickle

####################################################################
# Implementation of the patent algorithm to predict blood pressure
####################################################################




def get_blood_pressure(params, t13, p2p1, ts, td, normalized_ejection_area, age, height):         
    kt = params[0:3]
    ka = params[3:5]
    ku = params[5:7]

    y_pred_total = 0
    y_pred_total = (kt[0] * t13  + kt[1] * td + kt[2] * ts) + (ka[0] * normalized_ejection_area + ka[1] * p2p1) + (ku[0] * age + ku[1] * height)

    return y_pred_total


def sum_of_squares(params, df, Y, age, height):
    t13_array = np.transpose((df[['T13']]).to_numpy())
    p2p1_array = np.transpose((df[['P2P1']]).to_numpy())
    td_array = np.transpose((df[['TD']]).to_numpy())
    ej_area_array = np.transpose((df[['NEj Area']]).to_numpy())
    ts_array = np.transpose((df[['TS']]).to_numpy())

    y_pred = get_blood_pressure(params,np.mean(t13_array),  np.mean(p2p1_array), np.mean(ts_array), np.mean(td_array), np.mean(ej_area_array), age, height)
    error = np.sqrt(((y_pred - Y) ** 2).sum())

    print('error', error)
    return error


# get values for k that fir calibration value
def calculate_k_values(calibration_value, df, age, height):
    X = df
    Y = calibration_value

    kt_01 =kt_02 =kt_03 = ka_01 = ka_02 = ku_01 = ku_02 = 1

    res = least_squares(lambda x: sum_of_squares(x, X, Y, age, height), [kt_01, kt_02, kt_03 ,ka_01, ka_02, ku_01, ku_02])

    print('res', res)

    return res

    
# predict blood pressure value with known k values
def predict_blood_pressure(k_params_dia, k_params_sys, df, age, height):
    t13_array = np.transpose((df[['T13']]).to_numpy())
    p2p1_array = np.transpose((df[['P2P1']]).to_numpy())
    td_array = np.transpose((df[['TD']]).to_numpy())
    ej_area_array = np.transpose((df[['NEj Area']]).to_numpy())
    ts_array = np.transpose((df[['TS']]).to_numpy())

    y_pred_dia = get_blood_pressure(k_params_dia,np.mean(t13_array),  np.mean(p2p1_array), np.mean(ts_array), np.mean(td_array), np.mean(ej_area_array), age, height)
    print("diastolic blood pressure", y_pred_dia )
    y_pred_sys = get_blood_pressure(k_params_sys,np.mean(t13_array),  np.mean(p2p1_array), np.mean(ts_array), np.mean(td_array), np.mean(ej_area_array), age, height)
    print("systolic blood pressure", y_pred_sys )








