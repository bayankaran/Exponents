#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import pandas as pd1
import pandas as pd2
import numpy as np
import matplotlib.pyplot as plt
import csv
import nolds
from sklearn import preprocessing
# from scikit import preprocessing
import argparse
import random
import sys
import warnings
from pandas.tseries import offsets
from pandas.tseries.offsets import BDay

from datetime import date, timedelta, datetime
import datetime as dt
import pywt

from pandas.tseries import offsets

random.seed(10)

# for ease of running the script, only the instrument name
# as the argument on command
def convert_filename(filename):
    #new_filename = '/Users/sanjaydeshpande/Downloads/stocks/indicators/' + filename.upper() + '.csv'
    new_filename = '/Users/bayankaran/Workspace/MLE-Lyapunov/Instruments/' + filename.upper() + '.csv'
    return new_filename

def wt_transform(dataset):
    # Get wavelet coefficients
    coeffs = pywt.wavedec(dataset, 'db6', mode='symmetric',level=9,axis=-1)

    # Reconstruct the smoothed version of the time series from the decomposition
    approx = pywt.waverec(coeffs,'db6',mode='symmetric',axis=-1)
    return approx



# counting number of successes and failures
    # for the entire history of the instrument
    # and noting the price
    # returns true or false
def check_lexp_suc_fail(row_begin, row_end,filename):
    file_data_1 = pd1.read_csv(convert_filename(filename))

    # find low price in the window
    subset = file_data_1.iloc[row_begin:row_end]
    lowest_price = subset['Close'].min()
    print("Lowest price: " + str(round(lowest_price, 2)))

    if row_end < len(file_data_1):
        row_begin_data = file_data_1.iloc[row_begin]
        print(row_begin_data.name, row_begin_data.Date, round(row_begin_data.Close, 2))
        row_end_data = file_data_1.iloc[row_end]
        print(row_end_data.name, row_end_data.Date, round(row_end_data.Close, 2))

        # lowest price should be less than window open
        # also it should less than or equal to window close
        if (lowest_price < row_begin_data.Close) and (lowest_price <= row_end_data.Close):
            print("L.Exp success")
            return True
        else:
            print("L.exp fails")
            return False



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='Time series data file to read, in CSV format')
    parser.add_argument('--window', help='length of the time window', type=int, default=40)
    parser.add_argument('--delay', help='length of the time window', type=int, default=4)
    parser.add_argument('--col_name', help='The name of the column of data to use')
    parser.add_argument('--results', help='output file in CSV format')
    parser.add_argument('--graph', help='Do you want to generate graph True/False?', default=True)

    args = parser.parse_args()

    window_size = args.window
    delay = args.delay
    csv_columns = ['Date', 'Close']

    # Using col_name instead of arg.close
    col_name = 'Close'
    complete_file_name = convert_filename(args.name)
    tseries_df = pd.read_csv(complete_file_name)

    data_range = tseries_df.shape[0]

    print("The given data series has ", data_range, "rows")
    # series = tseries_df[args.col_name][0:data_range].to_numpy()
    series = tseries_df[col_name][0:data_range].to_numpy()
    print("The series length ", len(series))
    # Filter the noise using wavelet transform
    print('Calling wavelet transform...')
    approx=wt_transform(series)
    print()
    print(approx)
    series=approx

    # Compare the original signal with the reconstructed signal
    # plt.plot(approx,label = "wavelet_filtered_signal", color = "green")
    # plt.plot(tseries_df[col_name][0:data_range].to_numpy(),label = "original_signal", color = "blue")
    # plt.show()
    # print("Graph is displayed===================================================================================================================================")

    zero_line = []
    x_for_lle = []
    time_axis = []
    for i in range(data_range):
        time_axis.append(i)
        zero_line.append(0)
    i = 0
    mle = []

    while (i * delay) < (len(series) - window_size + delay) + 1:
        # print("*** Executing loop",i)
        window = series[i * delay: i * delay + window_size]
        lly = nolds.lyap_r(window, emb_dim=3, lag=1, min_tsep=None, tau=5, min_neighbors=10, trajectory_len=10,
                           fit=u'RANSAC', debug_plot=False, debug_data=False, plot_file=None, fit_offset=0)
        i += 1
        mle.append(lly)

    for j in range(i):
        x_for_lle.append((j * delay + window_size) - 1)

    price = []
    i = 0
    for i in x_for_lle:
        if (i >= data_range):
            price.append(series[i - delay])
        else:
            price.append(series[i])
    df = pd.DataFrame({'Time': x_for_lle, 'Lyapunov Exponent': mle, 'Price': price},
                      columns=['Time', 'Lyapunov Exponent', 'Price'])
    df.to_csv(args.results, index=False)

    # Produce analysis in text form
    # clear the screen
    print()
    print()
    print()
    print("The instances where the MLE fell below zero are:")
    print()
    sorted_df = df.sort_values('Lyapunov Exponent')
    tr_df = sorted_df[sorted_df['Lyapunov Exponent'] < 0]
    drops = []
    fail = 0
    success = 0

    for i in range(len(tr_df)):
        if (i + window_size + 1) < data_range:
            price_point = (tr_df.iloc[i]['Time']) + window_size
            next_price_point = (tr_df.iloc[i]['Time']) + window_size + delay
            print("Tick on Time Axis =", int(tr_df.iloc[i]['Time']))
            print("MLE=", tr_df.iloc[i]['Lyapunov Exponent'])
            print("Previous MLE Value=", tr_df.iloc[i - 1]['Lyapunov Exponent'])
            print("The next expected Price drop is between ", int(tr_df.iloc[i]['Time']), int(price_point - delay))
            row_up_range = int(tr_df.iloc[i]['Time'])
            row_low_range = int(price_point - delay)

            result = check_lexp_suc_fail(row_up_range, row_low_range,args.name)
            if not result:
                fail = fail + 1
            else:
                success = success + 1
            # END

            drops.append(int(price_point - delay))
            print()
    future_drops = [item for item in drops if item > data_range]
    future_drops.sort()

    def subtract_constant(numbers, value):
        return [num - value for num in numbers]

    print()
    print("=====================================================")
    print(args.name.upper() + " | Window " + str(args.window) + " | Delay " + str(args.delay))
    print("The given series had ", data_range, " number of days")

    print("Historical predictions: " + str(success + fail))

    success_rate = (success / (success + fail)) * 100
    failure_rate = (fail / (success + fail)) * 100

    print("Success Rate:", round(success_rate, 2), "%")
    print("Failure Rate:", round(failure_rate, 2), "%")
    print("The expected Price drop points in the future are: " + str(future_drops))

    def check_and_adjust_holidays(last_date, next_business_day, holidays):

        for date1 in pd.date_range(start=last_date, end=next_business_day - offsets.BDay(1)):
            if date1 in holidays:
                next_business_day = next_business_day + offsets.BDay(1)

        return next_business_day

    def find_next_business_days(future_day, filename):

        df1 = pd2.read_csv(filename)
        last_date = df1['Date'].iloc[-1]
        last_date = pd2.to_datetime(last_date)

        last_row_number = df1.shape[0]
        estimated_days = future_day - last_row_number

        next_business_day = last_date + BDay(estimated_days)

        holidays = [pd2.to_datetime('2024-05-27'), pd2.to_datetime('2024-06-19'),
                    pd2.to_datetime('2024-07-04'), pd2.to_datetime('2024-09-02'),
                    pd2.to_datetime('2024-11-28'), pd2.to_datetime('2024-12-25')]

        holiday_adjusted_business_day = check_and_adjust_holidays(last_date, next_business_day, holidays)

        return holiday_adjusted_business_day

    for drop_point in future_drops:
        future_day = find_next_business_days(drop_point, complete_file_name)
        print(drop_point, future_day.strftime('%Y-%m-%d'))

    print("====================================================")

    # if args.graph:
        # plt.plot(time_axis, normalized_series[0,], color='r', label='Normalized Price')
        # plt.plot(drops, normalized_series[0,], color='g', label='Detected by MLE')


if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    main()