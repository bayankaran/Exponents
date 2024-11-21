# This code is used for scanning the stocks, identify trending one's
# It is also used for validating effectiveness of Hurst exponent

# July 25,2024
# The code is extended to add H computations for current window. The current window calculations include 
# H, mean and trend but not success of failure. This block should be helpful for trading

# The analysis is also automated to an extent where the reasons for failure are added automatically.
# One type of analysis includes whether the failure happened because of weak trend, weak H or unknown reasons.
# The other type of analysis is to track the sudden drop/upsurge  in H, mean or trend over one block, last 
# two blocks or last three blocks. This is to help assess good and bad cues from H for trading

import numpy as np
import pandas as pd
import argparse
import nolds
import pywt
import random
import sys
import warnings
import math
from datetime import date
import logging
from twelvedata import TDClient
import time

random.seed(10)


def wt_transform(dataset):
    # Get wavelet coefficients
    coeffs = pywt.wavedec(dataset, 'db9', mode='symmetric', level=11, axis=-1)

    # Reconstruct the smoothed version of the time series from the decomposition
    approx = pywt.waverec(coeffs, 'db9', mode='symmetric', axis=-1)
    return approx


def modify_trend_column(data):
    # Access the first element of the list and round it to one decimal
    trend = np.around(data[0], decimals=2)
    return trend.flatten()[0]


def modify_hurst_column(data):
    return '{:.2f}'.format(data)


def main():
    API_KEY = "daf380ce795e45b5a16e52b2ca1be77f"
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='text file containing instrument names.')
    parser.add_argument('--stock_class', help='class of stocks, high vol, low vol etc.', default='-')
    parser.add_argument('--col_name', help='name of the column of data to use')
    parser.add_argument('--window', help='width of rolling window', default=10)
    parser.add_argument('--rows', help='no of days/rows, should greater than 300 ', default=4000)
    parser.add_argument('--history', help='print all past blocks', default=True)
    parser.add_argument('--export_type', help='only current window', default='all')

    logging.basicConfig(filename="newfile.log", format='%(asctime)s %(message)s', filemode='w')
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    args = parser.parse_args()
    col_name = args.col_name
    window = int(args.window)
    stock_class = args.stock_class

    stocks_days = []

    # datastructure to hold all the results
    results = {
        "block": 0,
        "stock": "",
        "start_date": "",
        "end_date": "",
        "mean_price": 0,
        "trend": [],
        "h": 0,
        "mean_vol": 0,
        "next_mean_price": 0,
        "success": True,
        "alt_success": True,
        "reason_for_failure": ""
    }
    df_results = pd.DataFrame(results)

    mem_counter = 0
    data_frame_var_name = ""

    def check_memory_footprint(df, counter, data_frame_var_name):
        print()
        print("---------------------------------")
        print("*** Counter", counter, ", data frame name", data_frame_var_name)
        print("*** DataFrame shape:", df.shape)
        print("*** DataFrame memory usage (bytes):\n", df.memory_usage(deep=True))
        print("*** Total memory usage (MB):", df.memory_usage(deep=True).sum() / (1024 * 1024))
        print()

        # Additional information
        print("\nData types:")
        print(df.dtypes)

    # AK
    # the following two methods
    # make sure we delete rows from beginning
    # and we calculate H until the last/most recent available date
    def find_closest_divisible(A, B):
        remainder = A % B
        if remainder == 0:
            return A  # Already divisible by B
        elif remainder > 0:
            return A - remainder
        else:
            return A + abs(remainder)

    def del_rows(df_delete, n):
        return df_delete.iloc[n:]  # Skip n rows starting from index 0 (excluding header)

    file_count = 0
    all_file_count = 0

    def fetch_data(name, rows):
        td = TDClient(apikey=API_KEY)
        timeseries = td.time_series(
            symbol=f"{name}",
            interval="1day",
            order="asc",
            outputsize=f"{rows}"
            # exchange="NSE"
        )
        return timeseries

    with open(args.name) as file:
        for line in file:

            stock_name = line.rstrip()

            try:
                ts = fetch_data(stock_name, args.rows)
            except Exception as e:
                print()
                print(stock_name)
                print("*** exception e:", e)
                print()
                return None

            df = ts.as_pandas()

            # this variable will be updated after we delete rows from the beginning
            # to get a block until the last day/most recent day
            rows = len(df)

            new_no_of_rows = 0
            rows_to_delete = 0

            logger.info(stock_name)
            logger.info(df)

            if rows % window != 0:
                new_no_of_rows = find_closest_divisible(rows, window)

            if new_no_of_rows >= 1:
                rows_to_delete = rows - new_no_of_rows

            # Delete rows and store the modified DataFrame
            if rows_to_delete > 0:
                df = del_rows(df.copy(), rows_to_delete)

            # AK
            # we don't care about specific dates
            # we focus on no of days of history
            start_date = df.index.min()
            end_date = df.index.max()
            # print("*** Begin Date: ", str(start_date))
            # print("*** End Date: ", str(end_date))

            # AK
            # after deleting rows in the beginning
            # again calculating how many rows...
            rows = len(df)
            logger.info(df)
            print(stock_name, rows)

            # this list updates key value of stock name and no of days/history
            stocks_days.append({stock_name: rows})

            tstart = 0  # read from the beginning
            tend = df.shape[0]  # - window  # read till the end of available data
            logger.info("Reading data for %s", line.rstrip())
            logger.info("The file %s has %d rows of data", line.rstrip(), df.shape[0])

            # print("*** tend: ", str(tend))

            # Create subset of data for the column needed (Open, Close, High, Low...)
            series = df[col_name][tstart:tend].to_numpy()
            data_size = len(series)
            # print("*** No of rows: ",data_size)
            logger.info("The %s column has %d entries", col_name, data_size)
            logger.info(series)

            # remove the noise by applying wavelet transform
            approx = wt_transform(series)
            logger.info("Wavelet transform is applied")
            logger.info(len(approx))
            logger.info(approx)

            # Hurst exponent
            # Calculates the Hurst exponent by a standard rescaled range (R/S) approach
            # The Hurst exponent is a measure for the “long-term memory” of a time series, meaning the long statistical 
            # dependencies in the data that do not originate from cycles.
            he = nolds.hurst_rs(approx, nvals=None, fit=u'RANSAC', debug_plot=False,
                                debug_data=False, plot_file=None, corrected=True, unbiased=True)
            logger.info("The Hurst exponent for entire series is %f", he)

            # initialize variables
            h = 0  # holds the Hurst exponent value for one block
            block = 0  # Given data series is broken in to number of blocks
            j = 0  # loop index to read all the elements in a block of data
            # iterate over all the blocks
            logger.info("Data size is %d")
            logger.info("The window size is %d", window)
            logger.info("The number of blocks are %d", math.floor(data_size / window))

            last_block = int(data_size / window)
            logger.info(last_block)

            # AK
            # we don't use math.floor, we can use last_block
            # because last_block will be final block
            # since its cleanly divisible last_block and math.floor will give same result.
            # while block < math.floor(data_size / window):
            while block < last_block:
                k = []
                r = 0
                logger.debug("r=%d", r)

                # iterate over all the elements within a block
                while r < window:
                    k.append(approx[j])
                    j = j + 1
                    r = r + 1
                logger.debug("%d block of data is ", block)
                logger.debug(k)
                # At this stage we have one block of data in variable k, ready to process
                # Calculate the Exponent for the block
                h = nolds.hurst_rs(k, nvals=None, fit=u'RANSAC', debug_plot=False,
                                   debug_data=False, plot_file=None, corrected=True, unbiased=True)
                logger.info("for block %d, the hurst exponent is %f", block, h)

                # Collect the results per block, Calculate the trend
                y = k
                x = []
                x.append(range(len(y)))  # Time variable
                x.append([1 for ele in range(len(y))])  # This adds the intercept, use range in Python3
                logger.debug("Calculating trend")
                logger.debug("x=")
                logger.debug(x)

                y = np.matrix(y).T
                x = np.matrix(x).T
                logger.debug("The matrices are")
                logger.debug(x)
                logger.debug(y)

                betas = ((x.T * x).I * x.T * y)
                logger.debug("Trend line Betas")
                logger.debug(betas)
                data_start = block * window
                data_end = data_start + window - 1
                next_start = data_end + 1
                next_end = next_start + window - 1
                logger.debug("The block of data is read from %d to %d from given time series", data_start, data_end)
                mean_price = np.round(df[col_name][data_start:data_end].to_numpy().mean(), 2)
                if block == last_block - 1:
                    next_mean_price = 0
                else:
                    next_mean_price = np.round(df[col_name][next_start:next_end].to_numpy().mean(), 2)
                logger.debug("The mean price for the %d is %f", block, mean_price)
                logger.debug("The mean price for the next block %d is %f", block + 1, next_mean_price)

                # Now, check the predictions. Based on H value, see if the next block of data follows predictions from
                # previous block of data
                success = False
                alt_success = False

                # Trending price movement
                if h > 0.5:
                    if (betas[0, 0] > 0):
                        # indicates price increase, positive trend
                        if block != last_block - 1:
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
                            if block != last_block - 1:
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
                # Find the reasons for failure. This is in reality finding weak trend and weak H
                reason = ""
                if block != last_block - 1:
                    if not success:
                        if abs(betas[0, 0]) < 0.1:
                            reason = "weak_trend"
                        else:
                            if (0.45 <= h <= 0.59):
                                reason = "weak_h"
                            else:
                                reason = "unknown"
                else:
                    reason = "Current Window"

                df2 = {
                    "block": block,
                    "stock": line.replace("\n", ""),
                    "mean_price": mean_price,
                    "trend": betas,
                    "h": h,
                    "mean_vol": np.round(df['volume'][data_start:data_end].to_numpy().mean(), 2),
                    "next_mean_price": next_mean_price,
                    "success": success,
                    "alt_success": alt_success,
                    "reason_for_failure": reason
                }
                logger.debug("The test results for block %d are", block)
                logger.debug(df2)
                pd.options.display.float_format = '{:.2f}'.format  # restrict the output to 2 decimal places
                df_results = df_results._append(df2, ignore_index=True)
                block = block + 1
            # print('*** No of files processed:', all_file_count)
            file_count = file_count + 1
            all_file_count = all_file_count + 1
            if file_count >= 40:
                file_count = 0
                print("*** Waiting 90 sec to use TD...")

                # df_temp_results = pd.DataFrame(df_results)
                # df_results = pd.DataFrame()

                # AK
                # checking mem of most probably the
                # largest data frame
                mem_counter = mem_counter + 1
                # check_memory_footprint(df_results, mem_counter, "df_results")
                time.sleep(90)

    df_details = pd.DataFrame(df_results)
    # df_details = pd.concat([df_temp_results, df_results], ignore_index=True)

    new_col_names = {'block': 'Block', 'stock': 'Stock',
                     'mean_price': 'Mean Price', 'trend': 'Trend', 'slope': 'Slope', 'h': 'Hurst',
                     'next_mean_price': 'Next Mean Price', 'success': 'Success', 'alt_success': "Alt. Success"}

    df_details = df_details.rename(columns=new_col_names)
    df_details = df_details.drop(columns=['start_date', 'end_date', 'mean_vol'])

    # AK
    # Apply functions to make Hurst and Trend readable, round etc.
    df_details['Hurst'] = df_details['Hurst'].apply(modify_hurst_column)
    df_details['Trend'] = df_details['Trend'].apply(modify_trend_column)

    filename = "h_det-" + stock_class + "-" + str(window) + ".csv"

    # AK
    # a new filename for filtered results
    # filtering for H > 0.65 only
    filename_filtered = "h_sel-" + str(window) + "-" + stock_class + "-" + str(date.today()) + ".csv"

    # AK
    # create a version of df_details
    # we are dropping columns and filtering results to show on H above 0.65
    df_filtered = df_details
    df_filtered_temp = df_filtered[df_filtered['reason_for_failure'] == 'Current Window']
    df_filtered_temp.drop(columns=['Next Mean Price', 'Success', 'Alt. Success', 'reason_for_failure'], inplace=True)
    df_filtered_temp.reset_index(drop=True)
    df_filtered_temp['Hurst'] = pd.to_numeric(df_filtered_temp['Hurst'], errors='coerce')
    df_filtered_temp = df_filtered_temp[df_filtered_temp['Hurst'] >= 0.65]

    # AK
    # This flag will check if filtered results to be printed
    # or full details
    # IMHO full details is too much stuff.
    if args.export_type != "all":
        df_filtered_temp.to_csv(filename_filtered, index=False)
    else:
        df_details.to_csv(filename, index=False)

    logger.info("*** The results dataframe")
    logger.info(df_results)

    # create subset of data for one stock
    # Create a grouped object by field stock
    df_per_stock = df_results.groupby('stock')
    logger.info("The results dataframe grouped by stock names")
    logger.info(df_per_stock)
    print(df_per_stock)
    # Calculate the summary report
    pd.options.display.float_format = '{:.2f}'.format
    print("Stock Name, SuccessRate, Alt SuccessRate")
    for stock, entries in df_per_stock:
        print(stock, ",", round(entries['success'].sum() / len(entries) * 100, 2), ",",
              round(entries['alt_success'].sum() / len(entries) * 100, 2))
        # logger.info(stock,"|",entries['success'].sum(),"|",round(entries['success'].sum()/len(entries)*100,2),"|")

    # Calculate for only trending stocks
    print("Success rate when we have h>0.5, trending")
    df_trend = df_results[df_results['h'] > 0.5]

    print("df_trend is subset of df_results, when h > 0.5")
    df_trend = df_trend.groupby('stock')
    pd.options.display.float_format = '{:.2f}'.format

    data_summary = []
    column_headers = ['Stock', 'History/Days', 'Window', 'Success', 'Alt Success']

    for st, entries in df_trend:
        print(st, ",", round(entries['success'].sum() / len(entries) * 100, 2), ",",
              round(entries['alt_success'].sum() / len(entries) * 100, 2))
        success_perc = int(round(entries['success'].sum() / len(entries) * 100, 2))
        alt_success_perc = int(round(entries['alt_success'].sum() / len(entries) * 100, 2))

        # AK
        # can't find a better way to keep no of history/days/rows
        # other than storing in a list/dictionary in the beginning
        # if history/days are printed straight from df, it prints multiple times
        history_days = ""
        for item in stocks_days:
            if list(item.keys())[0] == st:
                history_days = list(item.values())[0]
        data_summary.append([st, history_days, window, str(success_perc) + "%", str(alt_success_perc) + "%"])

    df_summary = pd.DataFrame(data_summary, columns=column_headers)
    filename = "h_summary-" + stock_class + "-" + str(window) + ".csv"
    df_summary.to_csv(filename, index=False)


if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    main()
