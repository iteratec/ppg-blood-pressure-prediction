import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import Calculation.key_point_calculation as key_point_calculation
from scipy.signal import find_peaks
from intersect import intersection
from scipy.signal import argrelextrema


def calculate_t13(systolic_peak, diastolic_peak):
    t13 = diastolic_peak[0] - systolic_peak[0]
    return t13


def calculate_normalized_ejection_area(notch, wave, time):
    nEjecA = None
    if notch != None:
            notch_x = notch[0]
            index = find_nearest(time, notch_x)
            ejection_span = wave[0: index + 1]
            time_span = time[0 : index + 1]
            # calculate the ejection area and the whole area under the curve
            ejection_area = np.trapz(ejection_span, x=time_span)
            whole_area = np.trapz(wave, x=time)

            # devides the ejection area by the whole area
            nEjecA = ejection_area / whole_area

    return nEjecA


def calculate_p2p1_ratio(systolic_peak, diastolic_peak):
    p2p1 = diastolic_peak[1] / systolic_peak[1]
    return p2p1


def calculate_cycle_duration(time, wave):
    tc = time[len(time)-1]
    return tc


def calculate_systolic_upstroke_time(time, systolic_peak):
    systolic_peak_x = systolic_peak[0]
    ts = systolic_peak_x - time[0]
    return ts


def calculate_diastolic_time(time, systolic_peak):
    systolic_peak_x = systolic_peak[0]
    td = time[len(time) - 1] - systolic_peak_x
    return td


def calculate_systolic_area(wave, time, systolic_peak):
    a_s = np.trapz(wave[0: systolic_peak + 1], x= time[0: systolic_peak + 1])
    '''
    plt.plot(time, wave)
    plt.fill_between(time[0: systolic_peak + 1], wave[0: systolic_peak + 1])
    plt.show()
    '''
    return a_s


def calculate_diastolic_area(wave, time, systolic_peak):
    a_d = np.trapz(wave[systolic_peak : len(wave)], x= time[systolic_peak : len(time)])
    '''
    plt.plot(time, wave)
    plt.fill_between(time[systolic_peak : len(time)], wave[systolic_peak : len(wave)])
    plt.show()
    '''
    return a_d



def calculate_width(wave, time, peakposition):
    width = None
    systolic_peak_y = wave[peakposition] / 2

    line_list = [systolic_peak_y] * len(time)

    x, y = intersection(time, wave, time, line_list)

    if len(x) > 1:
        width = x[1] - x[0]

    return width
    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_points_and_features(single_waves, min_size):
    # feature lists
    t13_list = []
    normalized_ejection_area_list = []
    p2p1_list = []
    tc_list = []
    ts_list = []
    td_list = []
    a_s_list = []
    a_d_list = []
    width_list = []

    
    for i in range(0, len(single_waves)): 
        # get single wave from list
        wave = single_waves[i][1]
        time = single_waves[i][0]

        # start time at 0
        time = time - min(time)

        # check wave has an appropriate length
        if max(time) > min_size:
            # calculate key points
            peakpositions, _ = find_peaks(wave, height=0.8)
            systolic_peak = [time[peakpositions[0]], wave[peakpositions[0]]]
            max_slope_point = key_point_calculation.detection_maximum_slope_point(wave, time, peakpositions[0])
            diastolic_peak = key_point_calculation.detection_diastolic_peak(wave, time, peakpositions[0])
            if diastolic_peak != None:
                notch = key_point_calculation.get_notch(wave, time, diastolic_peak[0], peakpositions[0])
                if notch != None:
                    inflection_point = key_point_calculation.get_inflection_point(wave, time, diastolic_peak[0], peakpositions[0], notch[0])

                    # plot key points

                    #plot_everything(diastolic_peak, systolic_peak, max_slope_point, inflection_point, notch, wave, time)

                
                    # calculate features
                    t13 = calculate_t13(systolic_peak, diastolic_peak)
                    t13_list.append(t13)               

                    normalized_ejection_area = calculate_normalized_ejection_area(notch, wave, time)
                    normalized_ejection_area_list.append(normalized_ejection_area)               

                    p2p1 = calculate_p2p1_ratio(systolic_peak, diastolic_peak)
                    p2p1_list.append(p2p1)              
                    

                    tc = calculate_cycle_duration(time, wave)
                    tc_list.append(tc)             
                    

                    ts = calculate_systolic_upstroke_time(time, systolic_peak)
                    ts_list.append(ts)   
                    

                    td = calculate_diastolic_time(time, systolic_peak)
                    td_list.append(td)          

                    a_s = calculate_systolic_area(wave, time, peakpositions[0])
                    a_s_list.append(a_s)          
                    

                    a_d = calculate_diastolic_area(wave, time, peakpositions[0])
                    a_d_list.append(a_d)          
                    

                    width = calculate_width(wave, time, peakpositions[0])
                    width_list.append(width)     
                       
                    

    return t13_list, normalized_ejection_area_list, p2p1_list, tc_list, ts_list, td_list, a_s_list, a_d_list, width_list  


# plots all key points
def plot_everything(diastolic_peak, systole_peak, max_slope_point, inflection_point, notch, wave, time):
    i_notch = find_nearest(time, notch[0] )
    i_dia = find_nearest(time, diastolic_peak[0])
    i_inf = find_nearest(time, inflection_point[0])
    plt.plot(time, wave)
    plt.title('Schlüsselpunkte')
    plt.xlabel('Time(ms)')
    plt.plot( systole_peak[0], systole_peak[1] , 'ro')
    plt.plot( max_slope_point[0], max_slope_point[1], 'ro')
    plt.plot( diastolic_peak[0], wave[i_dia] , 'ro')
    plt.plot( notch[0], wave[i_notch] , 'bo')
    plt.plot( inflection_point[0], wave[i_inf] , 'go')
    plt.show()
