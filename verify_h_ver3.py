# This code is used for scanning the stocks, identify trending one's
# It is also used for validating effectiveness of Hurst exponent
import numpy as np
import pandas as pd
import argparse
import nolds
import pywt
import random
import sys
import warnings
import yfinance as yf
import math
from datetime import date
import logging

random.seed(10)


def wt_transform(dataset):
    # Get wavelet coefficients
    # coeffs = pywt.wavedec(dataset, 'db6',mode='symmetric',level=9,axis=-1)
    coeffs = pywt.wavedec(dataset, 'db9', mode='symmetric', level=11, axis=-1)

    # Reconstruct the smoothed version of the time series from the decomposition
    # approx = pywt.waverec(coeffs,'db6',mode='symmetric',axis=-1)
    approx = pywt.waverec(coeffs, 'db9', mode='symmetric', axis=-1)
    return approx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='A text file that contains all the instrument names. One name per line')
    parser.add_argument('--results', help='where to store the detailed results, a filename')
    parser.add_argument('--col_name', help='The name of the column of data to use')
    parser.add_argument('--window', help='The width of rolling window', default=10)
    parser.add_argument('--start_dt', help='The starting date for the analysis', default='2021-01-01')
    parser.add_argument('--end_dt', help='The end date for the analysis', default='2024-07-17')

    logging.basicConfig(filename="newfile.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    args = parser.parse_args()
    end_date = args.end_dt  # end date for data retrieval
    start_date = args.start_dt  # Beginning date for our historical data retrieval
    col_name = args.col_name
    window = int(args.window)

    # datastructure to hold all the results
    results = {
        "block": 0,
        "stock": "",
        "mean_price": 0,
        "trend": [],
        "h": 0,
        "mean_vol": 0,
        "next_mean_price": 0,
        "success": True,
        "alt_success": True
    }
    df_results = pd.DataFrame(results)

    with open(args.name) as file:
        for line in file:
            print(line.rstrip())
            # Read the data from yfinance
            df = yf.download(line, start=start_date, end=end_date)  # Function used to fetch the data
            tstart = 0  # read from the beginning
            tend = df.shape[0] - window  # read till the end of available data
            print("Reading data for %s", line.rstrip())
            print("The file %s has %d rows of data", line.rstrip(), df.shape[0])

            # Create subset of data for the column needed (Open, Close, High, Low...)
            series = df[col_name][tstart:tend].to_numpy()
            data_size = len(series)
            print("The %s column has %d entries", col_name, data_size)
            print(series)
            if data_size < 300:
                print("Insufficient data to process")
                print("The data points are less than 300")
                logger.error("Insufficient data to process")
                exit(0)
            # remove the noise by applying wavelet transform
            approx = wt_transform(series)
            print("Wavelet transform is applied")
            print(approx)

            # Hurst exponent
            # Calculates the Hurst exponent by a standard rescaled range (R/S) approach
            # The Hurst exponent is a measure for the “long-term memory” of a time series, meaning the long statistical 
            # dependencies in the data that do not originate from cycles.
            he = nolds.hurst_rs(approx, nvals=None, fit=u'RANSAC', debug_plot=False,
                                debug_data=False, plot_file=None, corrected=True, unbiased=True)
            print("The Hurst exponent for entire series is %f", he)

            # initialize variales
            h = 0  # holds the Hurst exponent value for one block
            block = 0  # Given data series is broken in to number of blocks
            j = 0  # loop index to read all the elements in a block of data
            # iterate over all the blocks
            print("Data size is %d")
            print("The window size is %d", window)
            print("The number of blocks are %d", math.floor(data_size / window))
            while block < math.floor(data_size / window):
                k = []
                r = 0
                print("r=%d", r)
                # iterate over all the elements within a block
                while r < window:
                    k.append(approx[j])
                    j = j + 1
                    r = r + 1
                print("%d block of data is ", block)
                print(k)
                # At this stage we have one block of data in variable k, ready to process
                # Calculate the Hurst Exponent for the block
                h = nolds.hurst_rs(k, nvals=None, fit=u'RANSAC', debug_plot=False,
                                   debug_data=False, plot_file=None, corrected=True, unbiased=True)
                print("for block %d, the hurst exponent is %f", block, h)

                # Collect the results per block

                # Calculate the trend
                y = k
                x = []
                x.append(range(len(y)))  # Time variable
                x.append([1 for ele in range(len(y))])  # This adds the intercept, use range in Python3
                print("Calculating trend")
                print("x=")
                print(x)

                y = np.matrix(y).T
                x = np.matrix(x).T
                print("The matrices are")
                print(x)
                print(y)

                betas = ((x.T * x).I * x.T * y)
                print("Trend line Betas")
                print(betas)
                data_start = block * window
                data_end = data_start + window - 1
                next_start = data_end + 1
                next_end = next_start + window - 1
                print("The block of data is read from %d to %d from given time series", data_start, data_end)
                mean_price = np.round(df[col_name][data_start:data_end].to_numpy().mean(), 2)
                next_mean_price = np.round(df[col_name][next_start:next_end].to_numpy().mean(), 2)
                print("The mean price for the %d is %f", block, mean_price)
                print("The mean price for the next block %d is %f", block + 1, next_mean_price)

                # Now, check the predictions. Based on H value, see if the next block of data follows predictions from 
                # previous block of data

                success = False
                alt_success = False
                # Trending price movement
                if h > 0.5:
                    if (betas[0, 0] > 0):
                        # indicates price increase, positive trend
                        if df[col_name][next_start:next_end].to_numpy().max() > mean_price:
                            alt_success = True
                        else:
                            alt_success = False
                        if next_mean_price > mean_price:
                            success = True
                        else:
                            success = False
                    else:
                        if (betas[0, 0] < 0):
                            if df[col_name][next_start:next_end].to_numpy().min() < mean_price:
                                alt_success = True
                            else:
                                alt_success = False
                            if next_mean_price < mean_price:
                                success = True
                            else:
                                success = False
                else:
                    # Mean reverting price movement
                    if h < 0.5:
                        margin = mean_price * 5 / 100
                        if (mean_price - margin <= next_mean_price <= mean_price + margin):
                            success = True
                        else:
                            success = False

                df2 = {
                    "block": block,
                    "stock": line.replace("\n", ""),
                    "mean_price": mean_price,
                    "trend": betas,
                    "h": h,
                    "mean_vol": np.round(df['Volume'][data_start:data_end].to_numpy().mean(), 2),
                    "next_mean_price": next_mean_price,
                    "success": success,
                    "alt_success": alt_success
                }
                print("The test results for block %d are", block)
                print(df2)
                pd.options.display.float_format = '{:.2f}'.format  # restrict the output to 2 decimal places
                df_results = df_results._append(df2, ignore_index=True)
                block = block + 1

    print("Results dataframe", df_results)
    # saving the dataframe
    df_results.to_csv(args.results)
    print("The results dataframe")
    print(df_results)
    print("The results are stored into %s file", args.results)

    # create subset of data for one stock
    # Create a grouped object by field stock
    df_per_stock = df_results.groupby('stock')
    print("The results dataframe grouped by stock names")
    print(df_per_stock)
    # Calculate the summary report
    pd.options.display.float_format = '{:.2f}'.format
    print("Stock Name,SuccessRate,Alt SuccessRate")
    for stock, entries in df_per_stock:
        print(stock, ",", round(entries['success'].sum() / len(entries) * 100, 2), ",",
              round(entries['alt_success'].sum() / len(entries) * 100, 2))
        # print(stock,"|",entries['success'].sum(),"|",round(entries['success'].sum()/len(entries)*100,2),"|")

    # Calculate for only trending stocks
    print("Success rate when we have h>0.5, in other words, trending")
    df_trend = df_results[df_results['h'] > 0.5]

    print("df_trend is subset of df_results, when h > 0.5")
    df_trend = df_trend.groupby('stock')
    pd.options.display.float_format = '{:.2f}'.format
    for st, entries in df_trend:
        print(st, ",", round(entries['success'].sum() / len(entries) * 100, 2), ",",
              round(entries['alt_success'].sum() / len(entries) * 100, 2))
        # print(st,"|",entries['success'].sum(),"|",round(entries['success'].sum()/len(entries)*100,2),"|")


if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    main()
