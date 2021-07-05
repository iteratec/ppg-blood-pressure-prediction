import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import warnings
from scipy import signal




def preprocess_dataset(ppg, invert=True):
     seconds = int(len(ppg) / 125) 
     time = np.linspace(0, seconds * 1000 , num= len(ppg), endpoint=True)

     # get preprocessed data and new time for upsampled data
     working_data, upsampled_ppg, newtime = process_data(time, ppg, seconds, invert)

     # get separating points 
     low_peaks = calculate_wave_separation_points(upsampled_ppg, working_data)

     # get separated waves
     single_waves = get_single_wave_list(upsampled_ppg, low_peaks, working_data, newtime )

     return single_waves



# read data from csv and preprocess it
def process_data(time, ppg, seconds, invert = False):
     ppg_data = ppg

     # invert data (for O2Ring)
     if invert == True:
          ppg_data = hp.flip_signal(ppg, keep_range=True)


     # filter data lowpass
     filtered_ppg_low = hp.filter_signal(ppg_data, cutoff = 10, sample_rate = 125.0, order = 3, filtertype='lowpass')

     
     # interpolate data
     f = interp1d(time, filtered_ppg_low, kind='cubic')
     
     # upsamle data to 500Hz (from 3750 data points to 30s * 500Hz = 15000 data points)
     newtime = np.linspace(0, seconds * 1000, num=seconds * 500 , endpoint=True)
     upsampled_data = f(newtime)

     working_data, measures = hp.process(upsampled_data, 500.0)

     # filter data highpass
     filtered_ppg_high = hp.filter_signal(upsampled_data, cutoff = 0.75, sample_rate = 500.0, order = 3, filtertype='highpass')


     return working_data, filtered_ppg_high, newtime




# find all lows before a peak to seperate into single waves
def calculate_wave_separation_points(ppg, working_data): 

     peak_list = working_data['peaklist']
     rejected_peaks = working_data['removed_beats']

     # get all good peaks
     good_peaks = peak_list
     #[good_peaks.append(x) for x in peak_list if x not in rejected_peaks]

     low_peaks = []

     for i in range(len(good_peaks)-1):
          dataset = ppg[round(good_peaks[i]): round(good_peaks[i+1])]
          minima = argrelextrema(dataset, np.less)[0]

          if len(minima) != 0:
               values = dataset[minima]
               index = np.where(values == min(values))[0][0]
               low_point = minima[index] + good_peaks[i]
               if ppg[low_point] < 0:
                    low_peaks.append(low_point)
     '''
     plt.plot(ppg, 'b')
     round_to_whole_lowpeaks = [round(num) for num in low_peaks]
     plt.plot(low_peaks, ppg[round_to_whole_lowpeaks], 'go')
     plt.show()
     '''
     return low_peaks


# separates the data into single waves and puts them into a list
def get_single_wave_list(filtered_pgg, low_peaks, working_data, time):

     single_waves = []
     
     for i in range(1, len(low_peaks)-1):
           #get single wave
          low_peaks_start_end = low_peaks[i:i+2]
          single_wave = filtered_pgg[low_peaks_start_end[0]:low_peaks_start_end[1]]
          x_time = time[low_peaks_start_end[0] :low_peaks_start_end[1]]

          # move curve to 0 and normalize to 1
          one_wave = hp.scale_data(single_wave, lower=0, upper = 1)

          total_wave = [x_time, one_wave]
          single_waves.append(total_wave)

     return single_waves



# get the maximum slope point
def detection_maximum_slope_point(one_wave, time,  systolic_peak):
     
     # calculate scending section
     ascending_section_y = one_wave[0:systolic_peak] 
     ascending_section_x = time[0:systolic_peak]

     # get polynomial and first derivative
     polynomial = np.polyfit(ascending_section_x, ascending_section_y, 7)
     p = np.poly1d(polynomial)
     p1 = np.polyder(p)


     values_p1 = p1(range(int(max(ascending_section_x))))
     # get maximum value of the first derivative
     max_slope_point_x = np.where(values_p1==max(values_p1))[0][0]
     max_slope_point_y = p(max_slope_point_x)

     max_slope_point = [max_slope_point_x, max_slope_point_y ]

     return max_slope_point



# get the diastolic peak
def detection_diastolic_peak(one_wave, time, systolic_peak):

     diastolic_peak = None
     diastolic_peak_x = 0

     # get decending section
     descending_section_y = one_wave[systolic_peak+1: len(one_wave) -1] 
     descending_section_x = time[systolic_peak+1 : len(time)-1]

     # check for sufficient datapoints
     if len(descending_section_x) > 50:
          
          # get ploynomial (in paper order 7)
          try:
               polynomial = np.polyfit(descending_section_x, descending_section_y, 11)
          except np.RankWarning:
               try:
                    polynomial = np.polyfit(descending_section_x, descending_section_y, 9)
               except np.RankWarning:
                    return diastolic_peak

         
          p = np.poly1d(polynomial)

          # first and second derivative
          p1 = np.polyder(p)
          p2 = np.polyder(p1)
     
          # check how well the olynomial fits the data
          '''
          plt.plot(descending_section_x, descending_section_y, 'gx')
          plt.plot(descending_section_x, p(descending_section_x))
          plt.show()
          '''

          # get roots of first derivative
          roots = p1.r
          real_roots = roots[roots.imag == 0].real
          
          # get all values where the second derivative is negative
          real_roots_p2 = p2(real_roots) 
          x_min = real_roots[real_roots_p2 < 0]

          # check for value to be in certain area 
          if len(x_min) > 0:
               for i in range(len(x_min)):
                    if x_min[i] < descending_section_x[0] + 350 and x_min[i] > descending_section_x[0] + 150:
                         diastolic_peak_x = x_min[i]


          # if no value found take local minima of second derivative
          if diastolic_peak_x == 0:
               x_range = np.array(range(int(descending_section_x[0]), int(descending_section_x[len(descending_section_x) - 1])))
               p2_values = p2(x_range)
               minima_location = argrelextrema(p2_values, np.less_equal)[0] 
               minima = x_range[minima_location]

               sorted_minima = sorted(minima)

               if len(sorted_minima) > 0:
                    for i in range(len(sorted_minima)):
                         if sorted_minima[i] < descending_section_x[0] + 400 and sorted_minima[i] > descending_section_x[0] + 150:
                              diastolic_peak_x = sorted_minima[i]
                              break
                    if diastolic_peak_x == 0:
                         return diastolic_peak
               else:
                    plt.plot(time, one_wave)
                    plt.show()


          # get distolic peak x and y
          distolic_peak_y = p(diastolic_peak_x)
          diastolic_peak = [diastolic_peak_x , distolic_peak_y]

     return diastolic_peak



# calculate dicrotic notch
def get_notch(wave, time, diastolic_x, systolic_peak):

     notch = None
     notch_x = 0
     descending_section_y = wave[systolic_peak+1: len(wave) -1] 
     descending_section_x = time[systolic_peak+1 : len(time)-1]

     if len(descending_section_x) > 50:
          try:
               polynomial = np.polyfit(descending_section_x, descending_section_y, 11)
          except np.RankWarning:
               polynomial = np.polyfit(descending_section_x, descending_section_y, 9)
          
          p = np.poly1d(polynomial)
          p1 = np.polyder(p)
          p2 = np.polyder(p1)


          # get local maxima
          x_range = np.arange(int(descending_section_x[0]), int(descending_section_x[len(descending_section_x) - 1]), 0.1)
          p2_values = p2(x_range)
          maxima_location = argrelextrema(p2_values, np.greater)[0] 
          maxima = x_range[maxima_location]

          sorted_maxima = sorted(maxima)

          # check the x value is smaller as the diastolic x value
          if len(sorted_maxima) > 0:
               for i in range(len(sorted_maxima)):
                    if sorted_maxima[i] < diastolic_x:
                         notch_x = sorted_maxima[i]
               if notch_x == 0:
                    return notch

          notch_y = p(notch_x)
          notch_values = [notch_x, notch_y]

     return notch_values


# get the inflection point (between notch and diastolic peak)
def get_inflection_point(wave, time, diastolic_x, systolic_peak, notch_x):

     inflection_point = None
     inflection_point_x = 0
     descending_section_y = wave[systolic_peak+1: len(wave) -1] 
     descending_section_x = time[systolic_peak+1 : len(time)-1]

     if len(descending_section_x) > 50:

          polynomial = np.polyfit(descending_section_x, descending_section_y, 8)
          p = np.poly1d(polynomial)
          p1 = np.polyder(p)
          p2 = np.polyder(p1)


          roots = p2.r
          real_roots = roots[roots.imag == 0].real
          important_roots = []
          # check that roots are not double and in boundaries 
          [important_roots.append(x) for x in real_roots if x not in important_roots and x > descending_section_x[0] and x < max(descending_section_x)]

          # check for points smaller than diastolic x and larger then notch x
          if len(important_roots) > 0:
               for i in range(len(important_roots) - 1):
                    if important_roots[i] > notch_x and important_roots[i] < diastolic_x:
                         inflection_point_x = important_roots[i]

               if inflection_point_x == 0:
                    inflection_point_x = ( notch_x + diastolic_x ) / 2

          inflection_point_y = p(inflection_point_x)
          inflection_point = [inflection_point_x, inflection_point_y]

     return inflection_point