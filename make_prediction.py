import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import Calculation.key_point_calculation as key_point_calculation
import Calculation.feature_calculation as feature_calculation
import pandas as pd
from scipy import stats
import pickle
import Calculation.BP_prediction_functions as BP_prediction_functions

########################################################################
# parameters to modify
########################################################################

# user features
age = 25
height = 176

# read data from file
ppg = hp.get_data('Datasets/ppg_dataset3_2min.csv')

first_run=True


########################################################################
# get features from data
########################################################################


# get separated waves
single_waves = key_point_calculation.preprocess_dataset(ppg, True)


#calculate key points and features
t13, normalized_ejection_area, p2p1, tc, ts, td, a_s, a_d, width = feature_calculation.calculate_points_and_features(single_waves, 400)


# combine to dataframe
d = {'T13': t13, 'NEj Area': normalized_ejection_area, 'P2P1' : p2p1, 'TC': tc, 'TS' : ts, 'TD': td, 'AS' : a_s, 'AD' : a_d, 'Width' : width }

df = pd.DataFrame(data=d)
print(df)


########################################################################
# enable for first run
########################################################################

if(first_run):
    # calculate the parameters for diastolic blood pressure (DBP)
    result_dbp = BP_prediction_functions.calculate_k_values(72, df, age, height)
    params_dia = result_dbp['x']

    f = open('dbp.pckl', 'wb')
    pickle.dump(params_dia, f)
    f.close()

    # calculate the parameters for the systolic blood pressure (SBP)        
    result_sbp = BP_prediction_functions.calculate_k_values(111, df, age, height)
    params_sys = result_sbp['x']

    f = open('sbp.pckl', 'wb')
    pickle.dump(params_sys, f)
    f.close()

########################################################################
# enable for later runs
########################################################################

# read prediction parameters from file
f = open('dbp.pckl', 'rb')
params_dia = pickle.load(f)
f.close()

f = open('sbp.pckl', 'rb')
params_sys = pickle.load(f)
f.close()

BP_prediction_functions.predict_blood_pressure( params_dia, params_sys, df, age, height)